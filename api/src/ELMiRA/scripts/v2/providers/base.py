#!/usr/bin/env python3
"""
Abstract base class for MLLM providers.
All providers (OpenAI, Google) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator, Tuple
import numpy as np


@dataclass
class MLLMResponse:
    """Standardized response from any MLLM provider."""
    response_json: str
    success: bool
    error_message: str = ""
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult:
    """Standardized detection result."""
    label: str
    score: float
    center_x: float
    center_y: float
    width: float
    height: float
    bbox_raw: List[float] = field(default_factory=list)  # [x_min, y_min, x_max, y_max]


@dataclass  
class GroundedResponse:
    """Response with both actions and detections."""
    response_json: str
    detections: List[DetectionResult]
    success: bool
    error_message: str = ""
    latency_ms: float = 0.0


class BaseMLLMProvider(ABC):
    """Abstract base class for MLLM providers."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'openai', 'google')."""
        pass
    
    @abstractmethod
    def chat(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
    ) -> MLLMResponse:
        """
        Send a chat request to the MLLM.
        
        Args:
            prompt: Text prompt (user or system message)
            image: Optional BGR image array from camera
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Optional function/tool definitions
            
        Returns:
            MLLMResponse with JSON actions
        """
        pass
    
    @abstractmethod
    def chat_with_grounding(
        self,
        prompt: str,
        detect_objects: List[str],
        image: np.ndarray,
        temperature: float = 0.7,
    ) -> GroundedResponse:
        """
        Chat with built-in object grounding/detection.
        
        Args:
            prompt: Text prompt
            detect_objects: List of object names to detect
            image: BGR image array (required)
            temperature: Sampling temperature
            
        Returns:
            GroundedResponse with actions and detections
        """
        pass
    
    @abstractmethod
    def detect_objects(
        self,
        texts: List[str],
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[DetectionResult]:
        """
        Detect objects using MLLM vision.
        
        Args:
            texts: Object labels to detect
            image: BGR image array
            confidence_threshold: Minimum confidence
            
        Returns:
            List of DetectionResult
        """
        pass
    
    @abstractmethod
    def check_object_visibility(
        self,
        object_name: str,
        image: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Check if an object is visible in the image.
        
        Args:
            object_name: Name of object to find
            image: BGR image array
            
        Returns:
            Tuple of (is_visible, system_message)
        """
        pass
    
    def encode_image(self, image: np.ndarray, format: str = "jpeg") -> str:
        """Encode image to base64 string."""
        import cv2
        import base64
        
        if format == "jpeg":
            _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            _, buffer = cv2.imencode(".png", image)
        
        return base64.b64encode(buffer).decode("utf-8")
