"""English multi-turn GUI-agent prompt with left-click and termination tools."""

QWEN3_GUI_AGENT_MULTITURN_PROMPT = r"""You are a GUI Agent. You will receive an instruction, a chronological interaction history, and the current screen image. Complete the task through continuous multi-turn interaction rather than one-shot localization.

Tool definition:

{
  "type": "function",
  "function": {
    "name": "computer_use",
    "description": "Use a mouse to interact with a computer one action at a time. After each left_click, the environment returns an updated screen image for the next turn.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["left_click", "terminate"],
          "description": "The action to perform. Available actions:\n* left_click: Click the left mouse button once at a coordinate selected from the current screen image. Place the click near the center of the target's interactive area, then continue from the screen image returned in the next turn.\n* terminate: End the current task and report its completion status."
        },
        "coordinate": {
          "type": "array",
          "description": "The coordinate used by left_click, in the form [x, y]. Use the 0 to 1000 relative screen coordinate system: x increases from left to right and y increases from top to bottom."
        },
        "status": {
          "type": "string",
          "enum": ["success", "failure"],
          "description": "The task completion status."
        }
      },
      "required": ["action"]
    }
  }
}

Each turn performs one action. Inspect the current screen image and choose a tool using the interaction history and the visual state. After a left_click, the environment returns a new screen image and the next turn begins. Re-evaluate the updated interface before choosing the next action, and do not treat any turn as an independent one-shot localization task. Continue until the task is completed or cannot be completed.

The screen image associated with a historical round is the interface state before that round's action. The current-turn screen image is the interface state returned after the most recent action.

instruction:

{instruction}

{rounds}

Current turn:
{current_image}
"""


QWEN3_GUI_AGENT_MULTITURN_PROFILE = "qwen3_gui_agent_multiturn_v1"

_PROMPT_BODY_MARKER = "\n\ninstruction:\n"
if QWEN3_GUI_AGENT_MULTITURN_PROMPT.count(_PROMPT_BODY_MARKER) != 1:
    raise RuntimeError("multi-turn prompt must contain one dynamic body marker")

# The static portion is the system message. Each user turn supplies the
# instruction, its round label, and the image attached to that turn.
QWEN3_GUI_AGENT_MULTITURN_SYSTEM_PROMPT = (
    QWEN3_GUI_AGENT_MULTITURN_PROMPT.split(_PROMPT_BODY_MARKER, 1)[0].rstrip()
)


def qwen3_gui_agent_multiturn_user_text(instruction: str, *, round_index: int) -> str:
    """Render one causal turn after the static tool/rules system message."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
        raise ValueError("round_index must be a positive integer")
    return (
        f"instruction:\n{instruction.strip()}\n\n"
        f"Round {round_index}:\n"
        "Current turn:"
    )


__all__ = [
    "QWEN3_GUI_AGENT_MULTITURN_PROFILE",
    "QWEN3_GUI_AGENT_MULTITURN_PROMPT",
    "QWEN3_GUI_AGENT_MULTITURN_SYSTEM_PROMPT",
    "qwen3_gui_agent_multiturn_user_text",
]
