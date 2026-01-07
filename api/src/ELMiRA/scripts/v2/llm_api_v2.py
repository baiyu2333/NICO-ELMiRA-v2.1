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
        
        # Get API key from environment
        self.api_key = self._get_api_key()
        
        # Initialize provider
        self.provider: Optional[BaseMLLMProvider] = None
        self._init_provider()
        
        # Initialize image cache
        self.image_cache = CachedImageGrabber(
            topic=self.image_topic,
            cache_duration=self.cache_duration,
        )
        
        # Register v2 services (new names)
        rospy.Service("mllm_chat", PromptTextLLM, self.handle_chat)
        rospy.Service("mllm_vision", PromptVisionLLM, self.handle_vision)
        rospy.Service("mllm_visibility", CheckLLMObjectVisibility, self.handle_visibility)
        rospy.Service("mllm_detect", DetectWithMLLM, self.handle_detect)
        
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
            response = self.provider.chat(
                prompt=request.prompt,
                image=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Chat response in {latency_ms:.0f}ms")
            
            if response.success:
                rospy.loginfo(f"LLM output:\n{response.response_json}")
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
            )
            
            latency_ms = (time.time() - start_time) * 1000
            rospy.loginfo(f"Vision response in {latency_ms:.0f}ms")
            
            if response.success:
                rospy.loginfo(f"LLM output:\n{response.response_json}")
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
