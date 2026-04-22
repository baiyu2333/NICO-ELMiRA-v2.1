"""
ELMiRA shared LLM response parser — replaces 3 copy-pasted blocks.
"""

import json
import rospy


def parse_llm_actions(response_text: str) -> list:
    """
    Parse an LLM JSON response into a list of action dicts.
    
    Handles three formats:
    1. {"actions": [...]}       — standard multi-action
    2. {"action": "...", ...}   — single action (wrapped in list)
    3. anything else            — fallback to speak action
    
    Returns:
        list of action dicts, e.g. [{"action": "speak", "text": "..."}]
    """
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        rospy.logwarn(f"Failed to parse LLM response as JSON: {e}")
        return [{"action": "speak", "text": "Sorry, I had trouble understanding."}]

    if "actions" in parsed:
        return parsed["actions"]
    elif "action" in parsed:
        return [parsed]
    else:
        rospy.logwarn(f"LLM response missing 'actions' key: {list(parsed.keys())}")
        return [{"action": "speak", "text": response_text}]
