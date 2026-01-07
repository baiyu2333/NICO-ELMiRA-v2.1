#!/usr/bin/env python3
"""
Cached image grabber for efficient camera frame sharing.
Avoids multiple ROS wait_for_message calls.
"""

import time
import threading
from typing import Tuple, Optional
import numpy as np

import rospy
# import cv_bridge # Removed due to SystemError with numpy version mismatch
from sensor_msgs.msg import Image

def ros_image_to_numpy(msg):
    """
    Manual conversion of ROS Image message to Numpy array.
    Avoids cv_bridge dependency issues.
    Assumes bgr8 or rgb8 encoding.
    """
    dtype = np.uint8
    n_channels = 3
    
    if msg.encoding in ['bgr8', 'rgb8']:
        n_channels = 3
    elif msg.encoding in ['mono8']:
        n_channels = 1
        
    img_buf = np.frombuffer(msg.data, dtype=dtype)
    
    try:
        # Reshape according to height, width, channels
        img = img_buf.reshape((msg.height, msg.width, n_channels))
        
        # If encoding is rgb8 but we want bgr8 (for cv2), we swap.
        # But here we assume we generally work with BGR in this pipeline or don't care.
        # However, checking if msg.encoding == 'rgb8' and we want opencv format (BGR)
        if msg.encoding == 'rgb8':
             img = img[:, :, ::-1] # RGB to BGR
             
        return img
    except ValueError as e:
        rospy.logwarn(f"Image reshape failed: {e}. Buffer size: {len(img_buf)}, Expected: {msg.height * msg.width * n_channels}")
        return None

class CachedImageGrabber:
    """
    Caches the latest camera frame for efficient multi-consumer access.
    """
    
    def __init__(
        self,
        topic: str = "/nico/vision/right",
        cache_duration: float = 0.5,
        auto_subscribe: bool = True,
    ):
        self.topic = topic
        self.cache_duration = cache_duration
        
        # self._bridge = cv_bridge.CvBridge() # Removed
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0
        self._lock = threading.Lock()
        self._subscriber = None
        
        if auto_subscribe:
            self.start()
    
    def start(self):
        """Start background subscription."""
        if self._subscriber is None:
            self._subscriber = rospy.Subscriber(
                self.topic,
                Image,
                self._callback,
                queue_size=1,
                buff_size=2**24  # 16MB buffer for images
            )
            rospy.loginfo(f"CachedImageGrabber: Subscribed to {self.topic}")
    
    def stop(self):
        """Stop background subscription."""
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None
    
    def _callback(self, msg: Image):
        """Store latest frame."""
        try:
            # frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = ros_image_to_numpy(msg)
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._timestamp = time.time()
        except Exception as e:
            rospy.logwarn(f"CachedImageGrabber: Failed to convert image: {e}")
    
    def get_frame(self, max_age: float = None) -> np.ndarray:
        """
        Get the latest cached frame.
        
        Args:
            max_age: Maximum acceptable frame age in seconds.
                     If None, uses cache_duration.
                     If cached frame is older, waits for new one.
        
        Returns:
            BGR image as numpy array
        """
        max_age = max_age or self.cache_duration
        
        with self._lock:
            age = time.time() - self._timestamp
            if self._frame is not None and age < max_age:
                return self._frame.copy()
        
        # Cache miss - wait for new frame
        try:
            msg = rospy.wait_for_message(self.topic, Image, timeout=2.0)
            # frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = ros_image_to_numpy(msg)
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._timestamp = time.time()
                return frame
            else:
                raise ValueError("Converted frame is None")
        except Exception as e:
            # If we have a stale frame, return it
            with self._lock:
                if self._frame is not None:
                    rospy.logwarn(f"CachedImageGrabber: Timeout/Error, returning stale frame (age: {time.time() - self._timestamp:.1f}s)")
                    return self._frame.copy()
            
            # If completely no frame (e.g. no camera in VM), return dummy black image
            rospy.logwarn(f"CachedImageGrabber: Failed to get frame ({e}). Returning DUMMY image.")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Write "NO CAMERA" on the dummy image for debugging clarity
            try:
                import cv2
                cv2.putText(dummy_frame, "NO CAMERA", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            except ImportError:
                pass
                
            return dummy_frame
    
    def get_frame_with_timestamp(self) -> Tuple[np.ndarray, float]:
        """Get frame and its capture timestamp."""
        frame = self.get_frame()
        with self._lock:
            return frame, self._timestamp
    
    def has_recent_frame(self, max_age: float = None) -> bool:
        """Check if a recent frame is available without blocking."""
        max_age = max_age or self.cache_duration
        with self._lock:
            if self._frame is None:
                return False
            age = time.time() - self._timestamp
            return age < max_age


# Global instance for shared access
_global_grabber: Optional[CachedImageGrabber] = None


def get_cached_grabber(topic: str = "/nico/vision/right") -> CachedImageGrabber:
    """Get or create global cached image grabber."""
    global _global_grabber
    if _global_grabber is None:
        _global_grabber = CachedImageGrabber(topic)
    return _global_grabber
