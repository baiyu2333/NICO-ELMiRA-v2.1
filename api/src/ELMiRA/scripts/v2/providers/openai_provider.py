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
    
    DEFAULT_MODEL = "gpt-5.2"
    
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
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Build message array for API call."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if history:
            messages.extend(history)
            
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
        history: Optional[List[Dict]] = None,
    ) -> MLLMResponse:
        """Send chat request to GPT-4o."""
        start_time = time.time()
        
        try:
            messages = self._build_messages(prompt, image, history)
            
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
    
    @staticmethod
    def _draw_grid(image: np.ndarray, cols: int, rows: int,
                   labels: List[List[str]]) -> np.ndarray:
        """
        Draw a labeled grid overlay on the image.
        
        Args:
            image: BGR image array
            cols: number of columns
            rows: number of rows
            labels: 2D list of cell labels [row][col]
            
        Returns:
            Copy of image with grid drawn on it
        """
        import cv2
        
        img = image.copy()
        h, w = img.shape[:2]
        cell_w = w / cols
        cell_h = h / rows
        
        # Draw grid lines
        for c in range(1, cols):
            x = int(c * cell_w)
            cv2.line(img, (x, 0), (x, h), (0, 255, 0), 2)
        for r in range(1, rows):
            y = int(r * cell_h)
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 2)
        
        # Draw cell labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(h, w) / 600)
        thickness = max(1, int(font_scale * 2))
        
        for r in range(rows):
            for c in range(cols):
                label = labels[r][c]
                cx = int((c + 0.5) * cell_w)
                cy = int((r + 0.5) * cell_h)
                
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                tx = cx - text_size[0] // 2
                ty = cy + text_size[1] // 2
                
                # Background rectangle for readability
                pad = 4
                cv2.rectangle(
                    img,
                    (tx - pad, ty - text_size[1] - pad),
                    (tx + text_size[0] + pad, ty + pad),
                    (0, 0, 0), -1,
                )
                cv2.putText(img, label, (tx, ty), font, font_scale,
                            (0, 255, 0), thickness)
        
        return img

    def _grid_pass(
        self,
        image: np.ndarray,
        object_name: str,
        cols: int,
        rows: int,
        labels: List[List[str]],
    ) -> Optional[str]:
        """
        Ask GPT which grid cell contains the object.
        
        Returns the cell label (e.g., "B2") or None if not found.
        """
        grid_image = self._draw_grid(image, cols, rows, labels)
        
        # Flatten labels for the prompt
        all_labels = [lbl for row in labels for lbl in row]
        
        prompt = f"""This image has a grid overlay with labeled cells: {', '.join(all_labels)}.

Which cell contains the "{object_name}"? 

Rules:
- Pick the single cell whose label is closest to the CENTER of the object.
- If the object is not visible, respond with "none".
- Respond with ONLY this JSON format:
{{"cell": "LABEL", "confidence": 0.95}}

Example: {{"cell": "B2", "confidence": 0.9}}"""

        response = self.chat(prompt=prompt, image=grid_image, temperature=0.1)
        
        if not response.success:
            return None
        
        try:
            data = json.loads(response.response_json)
            cell = data.get("cell", "none")
            if cell.lower() == "none":
                return None
            # Validate that the cell is in our label set
            if cell in all_labels:
                return cell
            # Try case-insensitive match
            for lbl in all_labels:
                if lbl.lower() == cell.lower():
                    return lbl
            return None
        except (json.JSONDecodeError, KeyError):
            return None

    def _cell_to_coords(
        self, cell: str, cols: int, rows: int, labels: List[List[str]]
    ) -> Optional[tuple]:
        """
        Convert a cell label to normalized (center_x, center_y) coordinates.
        
        Returns (center_x, center_y) in [0, 1] range, or None if label not found.
        """
        for r in range(rows):
            for c in range(cols):
                if labels[r][c] == cell:
                    cx = (c + 0.5) / cols
                    cy = (r + 0.5) / rows
                    return (cx, cy)
        return None

    def detect_objects(
        self,
        texts: List[str],
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[DetectionResult]:
        """
        Detect objects using 2-pass grid-based visual grounding.
        
        Pass 1 (Coarse): Overlay a 4x3 grid (columns A-D, rows 1-3) on the
        full image, ask GPT which cell contains the object.
        
        Pass 2 (Fine): Crop the identified cell with padding, overlay a 3x3
        sub-grid, ask for the sub-cell to refine the position.
        
        This converts coordinate estimation from free-form number guessing
        (which GPT is bad at) into visual question answering (which GPT
        excels at), dramatically improving accuracy.
        """
        import cv2
        
        # --- Pass 1: Coarse grid (4 cols x 3 rows) ---
        coarse_cols, coarse_rows = 4, 3
        col_letters = ["A", "B", "C", "D"]
        coarse_labels = [
            [f"{col_letters[c]}{r+1}" for c in range(coarse_cols)]
            for r in range(coarse_rows)
        ]
        
        results = []
        
        for object_name in texts:
            # Pass 1: Which coarse cell?
            coarse_cell = self._grid_pass(
                image, object_name, coarse_cols, coarse_rows, coarse_labels
            )
            
            if coarse_cell is None:
                continue
            
            coarse_coords = self._cell_to_coords(
                coarse_cell, coarse_cols, coarse_rows, coarse_labels
            )
            if coarse_coords is None:
                continue
            
            coarse_cx, coarse_cy = coarse_coords
            
            # --- Pass 2: Fine grid on cropped cell ---
            h, w = image.shape[:2]
            cell_w = w / coarse_cols
            cell_h = h / coarse_rows
            
            # Find which column/row the coarse cell maps to
            coarse_col_idx = None
            coarse_row_idx = None
            for r in range(coarse_rows):
                for c in range(coarse_cols):
                    if coarse_labels[r][c] == coarse_cell:
                        coarse_col_idx = c
                        coarse_row_idx = r
                        break
            
            if coarse_col_idx is None:
                # Fallback: use coarse coordinates directly
                results.append(DetectionResult(
                    label=object_name,
                    score=0.7,
                    center_x=coarse_cx,
                    center_y=coarse_cy,
                    width=1.0 / coarse_cols,
                    height=1.0 / coarse_rows,
                ))
                continue
            
            # Crop with 50% padding on each side for context
            pad_x = cell_w * 0.5
            pad_y = cell_h * 0.5
            crop_x1 = max(0, int(coarse_col_idx * cell_w - pad_x))
            crop_y1 = max(0, int(coarse_row_idx * cell_h - pad_y))
            crop_x2 = min(w, int((coarse_col_idx + 1) * cell_w + pad_x))
            crop_y2 = min(h, int((coarse_row_idx + 1) * cell_h + pad_y))
            
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped.size == 0:
                results.append(DetectionResult(
                    label=object_name,
                    score=0.7,
                    center_x=coarse_cx,
                    center_y=coarse_cy,
                    width=1.0 / coarse_cols,
                    height=1.0 / coarse_rows,
                ))
                continue
            
            # Fine grid: 3x3 on the cropped region
            fine_cols, fine_rows = 3, 3
            fine_labels = [
                [f"{r+1}{c+1}" for c in range(fine_cols)]
                for r in range(fine_rows)
            ]
            
            fine_cell = self._grid_pass(
                cropped, object_name, fine_cols, fine_rows, fine_labels
            )
            
            if fine_cell is not None:
                fine_coords = self._cell_to_coords(
                    fine_cell, fine_cols, fine_rows, fine_labels
                )
                if fine_coords is not None:
                    fine_cx_local, fine_cy_local = fine_coords
                    
                    # Convert from crop-local coordinates to full-image coordinates
                    crop_w = crop_x2 - crop_x1
                    crop_h = crop_y2 - crop_y1
                    
                    abs_x = crop_x1 + fine_cx_local * crop_w
                    abs_y = crop_y1 + fine_cy_local * crop_h
                    
                    final_cx = abs_x / w
                    final_cy = abs_y / h
                    
                    results.append(DetectionResult(
                        label=object_name,
                        score=0.85,
                        center_x=final_cx,
                        center_y=final_cy,
                        width=crop_w / (w * fine_cols),
                        height=crop_h / (h * fine_rows),
                    ))
                    continue
            
            # Fallback to coarse if fine pass fails
            results.append(DetectionResult(
                label=object_name,
                score=0.7,
                center_x=coarse_cx,
                center_y=coarse_cy,
                width=1.0 / coarse_cols,
                height=1.0 / coarse_rows,
            ))
        
        return results
    
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
