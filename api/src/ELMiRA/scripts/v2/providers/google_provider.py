#!/usr/bin/env python3
"""
Google Gemini provider adapter for Gemini 2.5 Flash (Multimodal)
Implements BaseMLLMProvider interface using REST API.

Compatible with Python 3.8+ (no SDK dependency).
Uses direct REST API calls via httpx for maximum compatibility.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

# Add directory to path for absolute imports when run by ROS
_providers_dir = Path(__file__).parent.resolve()
if str(_providers_dir) not in sys.path:
    sys.path.insert(0, str(_providers_dir))

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from base import (
    BaseMLLMProvider,
    MLLMResponse,
    DetectionResult,
    GroundedResponse,
)


class GoogleProvider(BaseMLLMProvider):
    """
    Google Gemini 3 Flash provider using REST API.
    
    Supports multimodal inputs (text + images) for:
    - Chat/conversation with vision
    - Object detection with bounding boxes
    - Object visibility checking
    
    Uses Gemini's native bounding box output format (0-1000 scale).
    """
    
    # Gemini 3 Flash - multimodal model with vision capabilities
    DEFAULT_MODEL = "gemini-3-flash-preview"
    
    # API endpoint
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize Google Gemini provider.
        
        Args:
            api_key: Google AI API key. If None, reads from GOOGLE_API_KEY env var.
            model: Model name (default: gemini-2.5-flash)
            timeout: HTTP timeout in seconds
            **kwargs: Additional arguments passed to base class
        """
        resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        super().__init__(resolved_api_key, model or self.DEFAULT_MODEL, **kwargs)
        
        # HTTP client with connection pooling
        # Note: http2=False for Python 3.8 compatibility (h2 package issues)
        self.client = httpx.Client(
            timeout=timeout,
            http2=False,
        )
        
        self.system_prompt = self._build_system_prompt()
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for NICO robot interactions."""
        return """You are a child-size humanoid robot, named NICO, that interacts with a human user. Your task is to manipulate objects placed on a table at which you are seated, as well as communicating with the human. You will either receive a 'USER:' query with a transcription of the user's verbal input or a 'SYSTEM:' query whenever one of your systems returns feedback or status messages.

You need to respond with a list of actions to trigger your different systems to interact with the user. These actions will be processed and fully executed in sequence, before the user is prompted for input again or the interaction ends.

Speak: In order to verbally respond to the user, you should add a 'speak' action to the list of actions with an additional 'text' field. The text will be used to produce speech with your TTS module.

Describe: Whenever you need visual information to respond to a user query about your surroundings or objects on the table, you need to actively request it by adding a 'describe' action to the list of actions. This lets you look at the table and take an image with the right eye camera which you will receive as input.

Act: When instructed by the user to interact with objects on the table, you have to add the 'act' action, triggering your object detection and IK solver to produce physical actions with the left or right arm. You need to add a key for the 'type' of action and the target 'object' specified by the user for your systems to know which action to choose. Valid types are: 'touch' to touch the object with your hand, 'push' to move the object forward, 'push_left' to move it to the left, 'push_right' to move it to the right, 'show' to point towards it, 'grasp' to pick it up, and 'place' to put it down. If the user explicitly specifies a hand or arm, add "hand": "left" or "hand": "right" to the act action.

Quit: To end the interaction entirely, you should output the 'quit' signal with no additional parameters.

Try to be responsive and transparent to the user by adding 'speak' actions before or after executing other actions where appropriate.

Please always output your response as a valid JSON object containing the list of actions. Examples:
{"actions": [{"action": "speak", "text": "Sure, I can do that for you."}, {"action": "act", "object": "banana", "type": "touch"}]}
{"actions": [{"action": "describe"}]}
{"actions": [{"action": "speak", "text": "Goodbye! I hope we see each other again."}, {"action": "quit"}]}
{"actions": [{"action": "speak", "text": "I'll pick up the red ball for you."}, {"action": "act", "object": "red ball", "type": "grasp"}]}
{"actions": [{"action": "speak", "text": "I will use my left hand."}, {"action": "act", "object": "red object", "type": "grasp", "hand": "left"}]}"""

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 JPEG string.
        
        Args:
            image: BGR numpy array from OpenCV
            
        Returns:
            Base64-encoded JPEG string
        """
        import cv2
        
        # Encode as JPEG with quality 85 (good balance of quality/size)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    
    def _build_content_parts(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build content parts for API request.
        
        Args:
            prompt: Text prompt
            image: Optional BGR image as numpy array
            
        Returns:
            List of content part dictionaries
        """
        parts = []
        
        # Add image part first (if provided)
        if image is not None:
            image_data = self._encode_image(image)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            })
        
        # Add text part
        parts.append({"text": prompt})
        
        return parts
    
    def _build_request_body(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = True,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Build the full request body for generateContent API.
        
        Args:
            prompt: Text prompt
            image: Optional image
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            json_mode: Whether to request JSON output
            history: Optional conversation history
            
        Returns:
            Request body dictionary
        """
        contents = []
        
        # Add history if provided
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                content = msg.get("content", "")
                # Ensure content is string
                if isinstance(content, list):
                    # Attempt to extract text from list content (ignoring images in history for now)
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts)
                
                contents.append({
                    "role": role, 
                    "parts": [{"text": str(content)}]
                })
        
        # Add current turn
        contents.append({
            "role": "user",
            "parts": self._build_content_parts(prompt, image)
        })
        
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            # Safety settings - allow all content for robot control
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }
        
        # Request JSON response format
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        
        return body
    
    def _get_api_url(self, action: str = "generateContent") -> str:
        """Get the full API URL with API key."""
        return f"{self.API_BASE}/models/{self.model}:{action}?key={self.api_key}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def _make_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with retry logic.
        
        Args:
            body: Request body
            
        Returns:
            Response JSON dictionary
            
        Raises:
            httpx.HTTPStatusError: On API error responses
        """
        response = self.client.post(
            self._get_api_url(),
            json=body,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_text_from_response(self, response_data: Dict[str, Any]) -> str:
        """
        Extract text content from API response.
        
        Args:
            response_data: Raw API response
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If response format is unexpected
        """
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                # Check for blocked response
                if "promptFeedback" in response_data:
                    block_reason = response_data["promptFeedback"].get("blockReason", "UNKNOWN")
                    raise ValueError(f"Response blocked: {block_reason}")
                raise ValueError("No candidates in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise ValueError("No parts in response content")
            
            # Concatenate all text parts
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            return "".join(text_parts)
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format: {e}")
    
    def chat(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
    ) -> MLLMResponse:
        """
        Send chat request to Gemini with optional image.
        
        Args:
            prompt: Text prompt (can include USER: or SYSTEM: prefix)
            image: Optional BGR image as numpy array
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            tools: Optional tool definitions (not used currently)
            
        Returns:
            MLLMResponse with JSON response or error
        """
        start_time = time.time()
        
        try:
            # Build request
            body = self._build_request_body(
                prompt=prompt,
                image=image,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                history=history,
            )
            
            # Make API call
            response_data = self._make_request(body)
            
            # Extract text
            content = self._extract_text_from_response(response_data)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Validate JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return MLLMResponse(
                    response_json="",
                    success=False,
                    error_message=f"Invalid JSON response: {e}. Raw: {content[:200]}",
                    latency_ms=latency_ms,
                )
            
            return MLLMResponse(
                response_json=content,
                success=True,
                latency_ms=latency_ms,
            )
            
        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            error_body = e.response.text[:500] if e.response else str(e)
            return MLLMResponse(
                response_json="",
                success=False,
                error_message=f"API error {e.response.status_code}: {error_body}",
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return MLLMResponse(
                response_json="",
                success=False,
                error_message=str(e),
                latency_ms=latency_ms,
            )
    
    def chat_with_grounding(
        self,
        prompt: str,
        detect_objects: List[str],
        image: np.ndarray,
        temperature: float = 0.7,
    ) -> GroundedResponse:
        """
        Chat with Gemini including object detection/grounding.
        
        Uses Gemini's native bounding box output in 0-1000 scale.
        
        Args:
            prompt: User prompt
            detect_objects: List of object names to detect
            image: BGR image as numpy array
            temperature: Sampling temperature
            
        Returns:
            GroundedResponse with actions and detections
        """
        start_time = time.time()
        
        # Enhanced prompt for grounding
        grounding_prompt = f"""{prompt}

Also locate these objects in the image and provide bounding boxes:
Objects to detect: {', '.join(detect_objects)}

Include a 'detections' array in your JSON response:
{{"actions": [...], "detections": [{{"label": "object_name", "box_2d": [y_min, x_min, y_max, x_max], "confidence": 0.95}}]}}

IMPORTANT: box_2d uses coordinates in 0-1000 scale (not 0-1).
Format: [y_min, x_min, y_max, x_max] where 0=top/left and 1000=bottom/right."""

        try:
            response = self.chat(
                prompt=grounding_prompt,
                image=image,
                temperature=temperature,
            )
            
            if not response.success:
                return GroundedResponse(
                    response_json="",
                    detections=[],
                    success=False,
                    error_message=response.error_message,
                    latency_ms=response.latency_ms
                )
            
            data = json.loads(response.response_json)
            detections = []
            
            for det in data.get("detections", []):
                # Gemini uses [y_min, x_min, y_max, x_max] in 0-1000 scale
                box = det.get("box_2d", [0, 0, 1000, 1000])
                
                if len(box) >= 4:
                    y_min, x_min, y_max, x_max = [v / 1000.0 for v in box[:4]]
                else:
                    continue
                
                detections.append(DetectionResult(
                    label=det.get("label", "unknown"),
                    score=det.get("confidence", 0.5),
                    center_x=(x_min + x_max) / 2,
                    center_y=(y_min + y_max) / 2,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    bbox_raw=[x_min, y_min, x_max, y_max]
                ))
            
            latency_ms = (time.time() - start_time) * 1000
            
            return GroundedResponse(
                response_json=response.response_json,
                detections=detections,
                success=True,
                latency_ms=latency_ms
            )
            
        except json.JSONDecodeError as e:
            latency_ms = (time.time() - start_time) * 1000
            return GroundedResponse(
                response_json="",
                detections=[],
                success=False,
                error_message=f"JSON parse error: {e}",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return GroundedResponse(
                response_json="",
                detections=[],
                success=False,
                error_message=str(e),
                latency_ms=latency_ms
            )
    
    def detect_objects(
        self,
        texts: List[str],
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[DetectionResult]:
        """
        Detect objects in image using Gemini vision.
        
        Args:
            texts: List of object names/descriptions to detect
            image: BGR image as numpy array
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of DetectionResult with bounding boxes
        """
        prompt = f"""Detect the following objects in this image: {', '.join(texts)}

For each detected object, provide:
- label: the object name exactly as listed above
- box_2d: [y_min, x_min, y_max, x_max] in 0-1000 scale
- confidence: detection confidence (0.0-1.0)

Respond with JSON only:
{{"detections": [{{"label": "object_name", "box_2d": [y_min, x_min, y_max, x_max], "confidence": 0.9}}]}}

Only include objects you can clearly see with confidence >= {confidence_threshold}.
If an object is not visible, do not include it."""

        response = self.chat(prompt=prompt, image=image, temperature=0.3)
        
        if not response.success:
            return []
        
        try:
            data = json.loads(response.response_json)
            results = []
            
            for det in data.get("detections", []):
                conf = det.get("confidence", 0)
                
                if conf >= confidence_threshold:
                    box = det.get("box_2d", [0, 0, 1000, 1000])
                    
                    if len(box) >= 4:
                        y_min, x_min, y_max, x_max = [v / 1000.0 for v in box[:4]]
                    else:
                        continue
                    
                    results.append(DetectionResult(
                        label=det.get("label", "unknown"),
                        score=conf,
                        center_x=(x_min + x_max) / 2,
                        center_y=(y_min + y_max) / 2,
                        width=x_max - x_min,
                        height=y_max - y_min,
                        bbox_raw=[x_min, y_min, x_max, y_max]
                    ))
            
            return results
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return []
    
    def check_object_visibility(
        self,
        object_name: str,
        image: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Check if a specific object is visible in the image.
        
        Args:
            object_name: Name of object to check for
            image: BGR image as numpy array
            
        Returns:
            Tuple of (is_visible, system_message)
        """
        prompt = f"""Look at this image carefully.
Is there a "{object_name}" visible on the table or in the scene?

Respond with JSON:
{{"object_visible": true, "system_message": ""}}
or
{{"object_visible": false, "system_message": "explanation of why not visible"}}

Be accurate - only say true if you can clearly see the object."""

        response = self.chat(prompt=prompt, image=image, temperature=0.3)
        
        if not response.success:
            return False, f"SYSTEM: Vision check failed: {response.error_message}"
        
        try:
            data = json.loads(response.response_json)
            visible = data.get("object_visible", False)
            message = data.get("system_message", "")
            
            if not visible:
                return False, f"SYSTEM: {message}" if message else "SYSTEM: Object not visible"
            return True, ""
            
        except json.JSONDecodeError:
            return False, "SYSTEM: Failed to parse visibility response"
