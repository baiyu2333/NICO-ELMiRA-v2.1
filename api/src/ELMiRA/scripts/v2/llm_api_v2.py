#!/usr/bin/env python3
"""
ELMiRA v2 - Unified MLLM Gateway Node

This ROS node provides a unified interface to multiple MLLM providers
(OpenAI, Google) with support for:
- Text and vision chat
- Object grounding/detection
- Visibility checking
- Streaming responses
- Backward compatibility with v1 services
"""

import os
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add the v2 directory to Python path for imports when run directly by ROS
_v2_dir = Path(__file__).parent.resolve()
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))

import rospy
import numpy as np

from elmira.srv import (
    PromptTextLLM, PromptTextLLMResponse,
    PromptVisionLLM, PromptVisionLLMResponse,
    CheckLLMObjectVisibility, CheckLLMObjectVisibilityResponse,
    DetectObjects, DetectObjectsResponse,
    DetectWithMLLM, DetectWithMLLMResponse,
)
from elmira.msg import DetectedObject

# Import v2 components (absolute imports now that path is set up)
from providers import get_provider, BaseMLLMProvider
from utils.image_cache import CachedImageGrabber


class MLLMGateway:
    """
    Unified MLLM Gateway for ELMiRA v2.
    
    Provides ROS services for:
    - Chat (text and vision)
    - Object detection with MLLM
    - Object visibility checking
    - Backward compatible v1 services
    """
    
    
    def __init__(self):
        rospy.init_node("mllm_gateway")
        
        # Load configuration from ROS params
        self.provider_name = rospy.get_param("~provider", "openai")
        self.model = rospy.get_param("~model", None)  # Use provider default
        self.temperature = rospy.get_param("~temperature", 0.7)
        self.max_tokens = rospy.get_param("~max_tokens", 4096)
        self.timeout = rospy.get_param("~timeout", 30.0)
        self.image_topic = rospy.get_param("~image_topic", "/nico/vision/right")
        self.cache_duration = rospy.get_param("~cache_duration", 0.5)
        
        # Context preservation
        self.last_user_prompt = ""
        self.refinement_target_object = ""  # Set by state machine before refinement call
        
        # Get API key from environment
        self.api_key = self._get_api_key()
        
        # Initialize provider
        self.provider: Optional[BaseMLLMProvider] = None
        self._init_provider()
        
        # Initialize original image cache
        self.image_cache = CachedImageGrabber(
            topic=self.image_topic,
            cache_duration=self.cache_duration,
        )
        
        # Dedicated left eye cache for grasp refinement to avoid right-arm occlusion
        self.left_eye_cache = CachedImageGrabber(
            topic="/nico/vision/left",
            cache_duration=self.cache_duration,
        )
        
        # Conversation History
        self.conversation_history = []
        self.max_history = 20
        
        # Register v2 services (new names)
        rospy.Service("mllm_chat", PromptTextLLM, self.handle_chat)
        rospy.Service("mllm_vision", PromptVisionLLM, self.handle_vision)
        rospy.Service("mllm_visibility", CheckLLMObjectVisibility, self.handle_visibility)
        rospy.Service("mllm_detect", DetectWithMLLM, self.handle_detect)
        rospy.Service("mllm_refine_grasp", PromptVisionLLM, self.handle_refine_grasp)
        
        # Register backward-compatible v1 services
        # These are the same services but with legacy names
        rospy.Service("llm_chat", PromptTextLLM, self.handle_chat)
        rospy.Service("llm_vision", PromptVisionLLM, self.handle_vision)
        rospy.Service("llm_object_visibility", CheckLLMObjectVisibility, self.handle_visibility)
        
        rospy.loginfo(f"MLLM Gateway started with provider: {self.provider_name}")
        rospy.loginfo(f"Model: {self.provider.model if self.provider else 'N/A'}")
    
    def _get_api_key(self) -> str:
        """Get API key from environment or config file."""
        # Try environment variable first
        if self.provider_name == "openai":
            key = os.environ.get("OPENAI_API_KEY")
        elif self.provider_name == "google":
            key = os.environ.get("GOOGLE_API_KEY")
        else:
            rospy.logerr(f"Unknown provider: {self.provider_name}")
            raise ValueError(f"Unknown provider: {self.provider_name}")
            
        if key:
            return key
            
        # Fallback to config file
        try:
            config_path = os.path.expanduser("~/.elmira_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Check if provider matches (case-insensitive)
                saved_provider = config.get("provider", "").lower()
                if saved_provider == self.provider_name:
                    key = config.get("api_key")
                    if key:
                        rospy.loginfo(f"Loaded API key for {self.provider_name} from {config_path}")
                        return key
        except Exception as e:
            rospy.logwarn(f"Failed to load config file: {e}")
            
        rospy.logerr(f"{self.provider_name.upper()}_API_KEY not set in environment or config")
        raise ValueError(f"{self.provider_name.upper()}_API_KEY not set")
    
    def _init_provider(self):
        """Initialize the MLLM provider."""
        try:
            self.provider = get_provider(
                provider_name=self.provider_name,
                api_key=self.api_key,
                model=self.model,
            )
            rospy.loginfo(f"Initialized {self.provider.provider_name} provider")
        except Exception as e:
            rospy.logerr(f"Failed to initialize provider: {e}")
            raise
    
    def _get_image(self) -> np.ndarray:
        """Get current camera frame."""
        return self.image_cache.get_frame()
    
    def handle_chat(self, request) -> PromptTextLLMResponse:
        """
        Handle chat request (text only, no image).
        
        Service: mllm_chat / llm_chat (v1 compat)
        """
        rospy.loginfo(f"Chat request: {request.prompt[:100]}...")
        
        # Update last user prompt context - Only if it's a real user query (starts with USER:)
        if request.prompt.startswith("USER:"):
            # Strip "USER:" prefix for cleaner context
            self.last_user_prompt = request.prompt.replace("USER:", "").strip()
            rospy.loginfo(f"Context updated: '{self.last_user_prompt}'")
        
        start_time = time.time()
        
        try:
            # Pass history to provider
            response = self.provider.chat(
                prompt=request.prompt,
                image=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                history=self.conversation_history
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Chat response in {latency_ms:.0f}ms")
            
            if response.success:
                rospy.loginfo(f"LLM output:\n{response.response_json}")
                
                # Update history
                content = request.prompt
                if content.startswith("USER:"):
                    content = content.replace("USER:", "").strip()
                
                # Add "User" input (could be User or System message)
                self.conversation_history.append({
                    "role": "user",
                    "content": content
                })
                # Add Assistant Message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.response_json
                })
                
                # Trim history
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                        
                return PromptTextLLMResponse(response=response.response_json)
            else:
                rospy.logerr(f"Chat failed: {response.error_message}")
                # Return error as JSON for backward compatibility
                error_response = json.dumps({
                    "actions": [
                        {"action": "speak", "text": "Sorry, I encountered an error processing your request."}
                    ],
                    "error": response.error_message
                })
                return PromptTextLLMResponse(response=error_response)
                
        except Exception as e:
            rospy.logerr(f"Chat exception: {e}")
            error_response = json.dumps({
                "actions": [
                    {"action": "speak", "text": "Sorry, something went wrong."}
                ],
                "error": str(e)
            })
            return PromptTextLLMResponse(response=error_response)
    
    def handle_vision(self, request) -> PromptVisionLLMResponse:
        """
        Handle vision request (chat with current camera image).
        
        Service: mllm_vision / llm_vision (v1 compat)
        """
        rospy.loginfo("Vision request received")
        
        start_time = time.time()
        
        try:
            # Get current camera frame
            image = self._get_image()
            
            # Construct prompt using previous context if available
            if self.last_user_prompt:
                prompt = (
                    f"The user previously asked: '{self.last_user_prompt}'. "
                    f"Now you have the visual information. "
                    f"If the user asked for a description, use the 'speak' action to tell them what you see. "
                    f"If they asked to point/act, use the 'act' action. "
                    f"DO NOT use the 'describe' action again."
                )
            else:
                prompt = (
                    "Describe what you see on the table using the 'speak' action. "
                    "DO NOT use the 'describe' action again."
                )
            
            rospy.loginfo(f"Vision Prompt: {prompt}")

            response = self.provider.chat(
                prompt=prompt,
                image=image,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                history=self.conversation_history
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Vision response in {latency_ms:.0f}ms")
            
            if response.success:
                rospy.loginfo(f"LLM output:\n{response.response_json}")
                
                 # Update history with vision interaction
                 # We treat the synthesized prompt as the user input here for history consistency
                 # although slightly redundant, it keeps the narrative linear for the LLM
                self.conversation_history.append({
                    "role": "user",
                    "content": prompt
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.response_json
                })
                
                # Trim history
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                return PromptVisionLLMResponse(response=response.response_json)
            else:
                rospy.logerr(f"Vision failed: {response.error_message}")
                error_response = json.dumps({
                    "actions": [
                        {"action": "speak", "text": "Sorry, I had trouble seeing the table."}
                    ],
                    "error": response.error_message
                })
                return PromptVisionLLMResponse(response=error_response)
                
        except Exception as e:
            rospy.logerr(f"Vision exception: {e}")
            error_response = json.dumps({
                "actions": [
                    {"action": "speak", "text": "Sorry, I couldn't process the image."}
                ],
                "error": str(e)
            })
            return PromptVisionLLMResponse(response=error_response)
    
    def handle_visibility(self, request) -> CheckLLMObjectVisibilityResponse:
        """
        Handle object visibility check request.
        
        Service: mllm_visibility / llm_object_visibility (v1 compat)
        """
        rospy.loginfo(f"Visibility check for: {request.prompt}")
        
        start_time = time.time()
        
        try:
            # Get current camera frame
            image = self._get_image()
            
            # Check object visibility
            visible, message = self.provider.check_object_visibility(
                object_name=request.prompt,
                image=image,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Visibility check in {latency_ms:.0f}ms: visible={visible}")
            
            return CheckLLMObjectVisibilityResponse(
                object_visible=visible,
                system_message=message
            )
            
        except Exception as e:
            rospy.logerr(f"Visibility check exception: {e}")
            return CheckLLMObjectVisibilityResponse(
                object_visible=False,
                system_message=f"SYSTEM: Visibility check failed: {e}"
            )
    
    def handle_detect(self, request) -> DetectWithMLLMResponse:
        """
        Handle object detection request using MLLM.
        
        Service: mllm_detect
        
        This provides an alternative to OWLv2 detection using the MLLM's
        visual grounding capabilities.
        """
        rospy.loginfo(f"MLLM detection request for: {request.texts}")
        
        start_time = time.time()
        
        # Get confidence threshold from request or use default
        confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.5
        
        try:
            # Get current camera frame
            image = self._get_image()
            
            # Detect objects
            detections = self.provider.detect_objects(
                texts=list(request.texts),
                image=image,
                confidence_threshold=confidence_threshold,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"MLLM detection in {latency_ms:.0f}ms: {len(detections)} objects")
            
            # Convert to ROS message format
            ros_detections = []
            for det in detections:
                ros_det = DetectedObject()
                ros_det.label = det.label
                ros_det.score = det.score
                ros_det.center_x = det.center_x
                ros_det.center_y = det.center_y
                ros_det.width = det.width
                ros_det.height = det.height
                ros_detections.append(ros_det)
            
            return DetectWithMLLMResponse(
                objects=ros_detections,
                success=True,
                error_message="",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            rospy.logerr(f"MLLM detection exception: {e}")
            return DetectWithMLLMResponse(
                objects=[],
                success=False,
                error_message=str(e),
                latency_ms=latency_ms
            )
    
    def handle_refine_grasp(self, request) -> PromptVisionLLMResponse:
        """
        Handle grasp refinement request.
        
        Takes a fresh camera image (with the robot's hand visible near the
        target object) and asks GPT to estimate the correction offset needed
        to align the hand with the object.
        
        The target object name is read from the ROS param /mllm_refine_target.
        
        Service: mllm_refine_grasp
        Returns JSON: {"x_offset_meters": float, "y_offset_meters": float,
                       "confidence": float, "description": str}
        """
        # Get target object from ROS param (set by grasp_refinement state)
        target_object = rospy.get_param("/mllm_refine_target", "the object")
        rospy.loginfo(f"Grasp refinement request for: {target_object}")
        
        start_time = time.time()
        try:
            # Determine which eye to use based on parameter
            camera_eye = rospy.get_param("/mllm_refine_eye", "right").lower()
            hand_side = rospy.get_param("/mllm_refine_hand", "right").lower()
            if hand_side not in ("left", "right"):
                hand_side = "right"
            if camera_eye == "left":
                image = self.left_eye_cache.get_frame()
                eye_desc = "LEFT eye, which gives you a slightly angled, unblocked side-view"
            else:
                image = self.image_cache.get_frame()
                eye_desc = "main RIGHT eye (forehead camera)"
                
            prompt = (
                f"You are controlling a robot arm. You are looking through the robot's {eye_desc}. "
                f"The robot's {hand_side} mechanical hand is currently reaching forward horizontally, attempting to clamp the '{target_object}'. "
                f"Estimate how far the {hand_side} hand (the mechanical gripper) needs to move "
                f"to perfectly clamp the '{target_object}'. The hand does NOT need to move down, just horizontal alignment.\n\n"
                f"Coordinate system:\n"
                f"- x_offset_meters: positive = move hand FORWARD (push deeper into object), "
                f"negative = move hand BACKWARD (pull away from object)\n"
                f"- y_offset_meters: positive = move hand LEFT (from robot's perspective), "
                f"negative = move hand RIGHT\n\n"
                f"The table is about 50cm wide and 40cm deep from the robot's perspective. "
                f"Objects are typically within 5-15cm of the hand after the initial approach.\n\n"
                f"If the hand is already very close to the object (within 1-2cm), "
                f"set both offsets to 0.\n\n"
                f"Respond with ONLY this JSON format:\n"
                f'{{"x_offset_meters": 0.0, "y_offset_meters": 0.0, '
                f'"confidence": 0.8, "description": "brief explanation"}}'
            )
            
            response = self.provider.chat(
                prompt=prompt,
                image=image,
                temperature=0.2,  # Low temperature for precision
                max_tokens=256,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Grasp refinement response in {latency_ms:.0f}ms")
            
            if response.success:
                rospy.loginfo(f"Refinement output: {response.response_json}")
                
                # Validate and clamp the offsets
                try:
                    data = json.loads(response.response_json)
                    x_off = float(data.get("x_offset_meters", 0.0))
                    y_off = float(data.get("y_offset_meters", 0.0))
                    
                    # Clamp to ±5cm to prevent wild corrections
                    MAX_CORRECTION = 0.05
                    x_off = max(-MAX_CORRECTION, min(MAX_CORRECTION, x_off))
                    y_off = max(-MAX_CORRECTION, min(MAX_CORRECTION, y_off))
                    
                    clamped_response = json.dumps({
                        "x_offset_meters": round(x_off, 4),
                        "y_offset_meters": round(y_off, 4),
                        "confidence": data.get("confidence", 0.5),
                        "description": data.get("description", ""),
                    })
                    rospy.loginfo(f"Clamped refinement: {clamped_response}")
                    return PromptVisionLLMResponse(response=clamped_response)
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    rospy.logwarn(f"Failed to parse refinement response: {e}")
                    # Return zero correction on parse failure
                    fallback = json.dumps({
                        "x_offset_meters": 0.0,
                        "y_offset_meters": 0.0,
                        "confidence": 0.0,
                        "description": f"Parse error: {e}",
                    })
                    return PromptVisionLLMResponse(response=fallback)
            else:
                rospy.logerr(f"Refinement failed: {response.error_message}")
                fallback = json.dumps({
                    "x_offset_meters": 0.0,
                    "y_offset_meters": 0.0,
                    "confidence": 0.0,
                    "description": f"API error: {response.error_message}",
                })
                return PromptVisionLLMResponse(response=fallback)
                
        except Exception as e:
            rospy.logerr(f"Grasp refinement exception: {e}")
            fallback = json.dumps({
                "x_offset_meters": 0.0,
                "y_offset_meters": 0.0,
                "confidence": 0.0,
                "description": f"Exception: {e}",
            })
            return PromptVisionLLMResponse(response=fallback)
    
    def run(self):
        """Run the gateway node."""
        rospy.loginfo("MLLM Gateway running...")
        rospy.spin()
        
        # Cleanup
        self.image_cache.stop()
        rospy.loginfo("MLLM Gateway shutdown complete")


def main():
    try:
        gateway = MLLMGateway()
        gateway.run()
    except Exception as e:
        rospy.logerr(f"MLLM Gateway failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
