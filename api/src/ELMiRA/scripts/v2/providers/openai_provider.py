#!/usr/bin/env python3
"""
OpenAI provider adapter for GPT-4o / GPT-4.5 / GPT-5
Implements BaseMLLMProvider interface.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Add directory to path for absolute imports when run by ROS
_providers_dir = Path(__file__).parent.resolve()
if str(_providers_dir) not in sys.path:
    sys.path.insert(0, str(_providers_dir))

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from base import (
    BaseMLLMProvider,
    MLLMResponse,
    DetectionResult,
    GroundedResponse,
)


class OpenAIProvider(BaseMLLMProvider):
    """OpenAI GPT-5.2 provider implementation."""
    
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(
        self,
        api_key: str,
        model: str = None,
        organization: str = None,
        **kwargs
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, **kwargs)
        self.client = OpenAI(api_key=api_key, organization=organization)
        
        # System prompt for NICO robot
        self.system_prompt = self._build_system_prompt()
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for NICO robot interactions."""
        return """You are a child-size humanoid robot, named NICO, that interacts with a human user. Your task is to manipulate objects placed on a table at which you are seated, as well as communicating with the human. You will either receive a 'USER:' query with a transcription of the user's verbal input or a 'SYSTEM:' query whenever one of your systems returns feedback or status messages.

You need to respond with a list of actions to trigger your different systems to interact with the user. These actions will be processed and fully executed in sequence, before the user is prompted for input again or the interaction ends.

Speak: In order to verbally respond to the user, you should add a 'speak' action to the list of actions with an additional 'text' field. The text will be used to produce speech with your TTS module.

Describe: ONLY use this when the user asks "what do you see?" or "describe the scene". Do NOT use this if the user asks you to point, touch, or grab something.

Act: When the user asks you to point at, touch, grasp, or move an object, use this action DIRECTLY. You do NOT need to 'describe' first; the 'act' command will automatically trigger the camera and detection system to find the object. Valid types: 'touch', 'push', 'push_left', 'push_right', 'show' (for pointing), 'grasp', 'place'.

Quit: To end the interaction entirely, you should output the 'quit' signal with no additional parameters.

Try to be responsive and transparent to the user by adding 'speak' actions before or after executing other actions where appropriate.

Please always output your response as a valid JSON object containing the list of actions. Examples:
{"actions": [{"action": "speak", "text": "Sure, I can do that for you."}, {"action": "act", "object": "banana", "type": "touch"}]}
{"actions": [{"action": "describe"}]}
{"actions": [{"action": "speak", "text": "Goodbye! I hope we see each other again."}, {"action": "quit"}]}
{"actions": [{"action": "speak", "text": "I'll pick up the red ball for you."}, {"action": "act", "object": "red ball", "type": "grasp"}]}"""

    def _build_messages(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Build message array for API call."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if image is not None:
            base64_image = self.encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict]] = None,
    ) -> MLLMResponse:
        """Send chat request to GPT-4o."""
        start_time = time.time()
        
        try:
            messages = self._build_messages(prompt, image)
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            # Validate JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return MLLMResponse(
                    response_json="",
                    success=False,
                    error_message=f"Invalid JSON response: {e}",
                    latency_ms=latency_ms,
                    raw_response={"content": content}
                )
            
            return MLLMResponse(
                response_json=content,
                success=True,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return MLLMResponse(
                response_json="",
                success=False,
                error_message=str(e),
                latency_ms=latency_ms
            )
    
    def chat_with_grounding(
        self,
        prompt: str,
        detect_objects: List[str],
        image: np.ndarray,
        temperature: float = 0.7,
    ) -> GroundedResponse:
        """Chat with object grounding using GPT-4o vision."""
        start_time = time.time()
        
        # Enhanced prompt for grounding
        grounding_prompt = f"""{prompt}

Additionally, locate the following objects in the image and provide their bounding boxes:
Objects to detect: {', '.join(detect_objects)}

Include a 'detections' array in your response with format:
{{"detections": [{{"label": "object_name", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95}}]}}

Coordinates should be normalized (0-1) relative to image dimensions."""

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
            
            # Parse detections from response
            data = json.loads(response.response_json)
            detections = []
            
            if "detections" in data:
                for det in data["detections"]:
                    bbox = det.get("bbox", [0, 0, 1, 1])
                    x_min, y_min, x_max, y_max = bbox
                    
                    detections.append(DetectionResult(
                        label=det.get("label", "unknown"),
                        score=det.get("confidence", 0.5),
                        center_x=(x_min + x_max) / 2,
                        center_y=(y_min + y_max) / 2,
                        width=x_max - x_min,
                        height=y_max - y_min,
                        bbox_raw=bbox
                    ))
            
            latency_ms = (time.time() - start_time) * 1000
            
            return GroundedResponse(
                response_json=response.response_json,
                detections=detections,
                success=True,
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
        """Detect objects using GPT-4o vision."""
        prompt = f"""You are an object detection system. Detect these objects in the image: {', '.join(texts)}

CRITICAL: Return PRECISE bounding box coordinates. The image uses normalized coordinates where:
- (0, 0) is the TOP-LEFT corner
- (1, 1) is the BOTTOM-RIGHT corner
- x increases from LEFT to RIGHT
- y increases from TOP to BOTTOM

For each detected object, provide:
- label: exact object name from the list
- bbox: [x_min, y_min, x_max, y_max] as normalized coordinates (0.0 to 1.0)
- confidence: detection confidence (0.0 to 1.0)

Be VERY precise with bounding box coordinates - they will be used for robotic manipulation.
The bbox should tightly enclose just the object, not the surrounding area.

Respond with ONLY this JSON format:
{{"detections": [{{"label": "object_name", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95}}]}}

Only include objects you can clearly see with confidence >= {confidence_threshold}."""

        response = self.chat(prompt=prompt, image=image, temperature=0.3)
        
        if not response.success:
            return []
        
        try:
            data = json.loads(response.response_json)
            results = []
            
            for det in data.get("detections", []):
                if det.get("confidence", 0) >= confidence_threshold:
                    bbox = det.get("bbox", [0, 0, 1, 1])
                    x_min, y_min, x_max, y_max = bbox
                    
                    results.append(DetectionResult(
                        label=det["label"],
                        score=det["confidence"],
                        center_x=(x_min + x_max) / 2,
                        center_y=(y_min + y_max) / 2,
                        width=x_max - x_min,
                        height=y_max - y_min,
                        bbox_raw=bbox
                    ))
            
            return results
            
        except (json.JSONDecodeError, KeyError):
            return []
    
    def check_object_visibility(
        self,
        object_name: str,
        image: np.ndarray,
    ) -> Tuple[bool, str]:
        """Check if object is visible in image."""
        prompt = f"""Is there a {object_name} visible on the table in this image?

Respond with JSON:
{{"object_visible": true/false, "system_message": "explanation if not visible"}}"""

        response = self.chat(prompt=prompt, image=image, temperature=0.3)
        
        if not response.success:
            return False, f"SYSTEM: Vision check failed: {response.error_message}"
        
        try:
            data = json.loads(response.response_json)
            visible = data.get("object_visible", False)
            message = data.get("system_message", "")
            
            if not visible:
                return False, f"SYSTEM: {message}"
            return True, ""
            
        except json.JSONDecodeError:
            return False, "SYSTEM: Failed to parse visibility response"
