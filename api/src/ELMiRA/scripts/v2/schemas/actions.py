#!/usr/bin/env python3
"""
Action schema definitions for ELMiRA v2.
These match the action types expected by the state machine and action_parser.
"""

from typing import Optional, Union, Literal, List
from dataclasses import dataclass, field, asdict
import json


@dataclass
class SpeakAction:
    """Speak action - robot says something to the user."""
    action: Literal["speak"] = "speak"
    text: str = ""
    
    def to_dict(self) -> dict:
        return {"action": self.action, "text": self.text}


# Valid action types for ActAction.interaction field
VALID_INTERACTION_TYPES = (
    "touch",       # Touch object with hand
    "push",        # Push object forward
    "push_left",   # Push object to the left
    "push_right",  # Push object to the right
    "show",        # Point at object
    "grasp",       # Pick up object (bimanual support)
    "place",       # Put down object (bimanual support)
    "open_hand",   # Open gripper
    "close_hand",  # Close gripper
)


@dataclass
class ActAction:
    """Act action - robot performs physical interaction with an object.
    
    Supports bimanual manipulation with both left and right arms.
    Arm selection is automatic based on object position (Y coordinate).
    
    Valid interaction types:
    - touch: Touch object with hand
    - push: Push object forward
    - push_left: Push object to the left
    - push_right: Push object to the right  
    - show: Point at object
    - grasp: Pick up object (includes hand close)
    - place: Put down object (includes hand open)
    - open_hand: Open gripper only
    - close_hand: Close gripper only
    """
    action: Literal["act"] = "act"
    target_object: str = ""
    interaction: str = ""  # One of VALID_INTERACTION_TYPES
    hand: Optional[Literal["left", "right"]] = None
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_object": self.target_object,
            "interaction": self.interaction,
            "hand": self.hand,
        }
    
    @property
    def is_valid_interaction(self) -> bool:
        """Check if interaction type is valid."""
        return self.interaction in VALID_INTERACTION_TYPES


@dataclass
class DescribeAction:
    """Describe action - robot describes what it sees in the scene."""
    action: Literal["describe"] = "describe"
    description: str = ""
    
    def to_dict(self) -> dict:
        return {"action": self.action, "description": self.description}


@dataclass
class QuitAction:
    """Quit action - end the conversation."""
    action: Literal["quit"] = "quit"
    farewell: str = ""
    
    def to_dict(self) -> dict:
        return {"action": self.action, "farewell": self.farewell}


@dataclass 
class ActionResponse:
    """
    Unified action response from MLLM.
    Wraps one of the specific action types.
    """
    actions: List[Union[SpeakAction, ActAction, DescribeAction, QuitAction]] = field(default_factory=list)
    raw_response: str = ""
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    
    def to_json(self) -> str:
        """Convert to JSON string for ROS service response."""
        actions_list = [a.to_dict() for a in self.actions]
        return json.dumps({"actions": actions_list}, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str, raw_response: str = "", 
                  latency_ms: float = 0.0, provider: str = "", 
                  model: str = "") -> "ActionResponse":
        """Parse MLLM JSON response into ActionResponse."""
        try:
            data = json.loads(json_str)
            actions = []
            
            actions_data = data.get("actions", [data]) if isinstance(data, dict) else [data]
            if isinstance(data, dict) and "action" in data and "actions" not in data:
                actions_data = [data]
            
            for action_data in actions_data:
                action_type = action_data.get("action", "")
                
                if action_type == "speak":
                    actions.append(SpeakAction(
                        text=action_data.get("text", action_data.get("speech", ""))
                    ))
                elif action_type == "act":
                    actions.append(ActAction(
                        target_object=action_data.get("target_object", action_data.get("object", "")),
                        interaction=action_data.get("interaction", action_data.get("type", "")),
                        hand=action_data.get("hand", action_data.get("side", action_data.get("arm"))),
                    ))
                elif action_type == "describe":
                    actions.append(DescribeAction(
                        description=action_data.get("description", action_data.get("text", ""))
                    ))
                elif action_type == "quit":
                    actions.append(QuitAction(
                        farewell=action_data.get("farewell", action_data.get("text", ""))
                    ))
            
            return cls(
                actions=actions,
                raw_response=raw_response,
                latency_ms=latency_ms,
                provider=provider,
                model=model
            )
        except json.JSONDecodeError as e:
            # Return empty response on parse error
            return cls(
                actions=[],
                raw_response=raw_response,
                latency_ms=latency_ms,
                provider=provider,
                model=model
            )
    
    @property
    def primary_action(self) -> Optional[Union[SpeakAction, ActAction, DescribeAction, QuitAction]]:
        """Get the first/primary action."""
        return self.actions[0] if self.actions else None
    
    @property
    def action_type(self) -> str:
        """Get the type of the primary action."""
        if self.primary_action:
            return self.primary_action.action
        return ""
