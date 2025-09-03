import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Cache for loaded tasks
_tasks_cache: Optional[Dict[str, Dict]] = None


def load_banana_tasks() -> Dict[str, Dict]:
    """Load banana task presets from presets/banana-tasks.json."""
    global _tasks_cache
    if _tasks_cache is not None:
        return _tasks_cache

    # Get the directory of this file
    current_dir = Path(__file__).parent
    tasks_path = current_dir / "presets" / "banana-tasks.json"
    
    if not tasks_path.exists():
        logger.error(f"banana-tasks.json not found at {tasks_path}")
        return {}

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Map by task name
            _tasks_cache = {
                entry["task"]: entry 
                for entry in data 
                if isinstance(entry, dict) and entry.get("task")
            }
            logger.info(f"Loaded {len(_tasks_cache)} banana tasks")
            return _tasks_cache
    except Exception as e:
        logger.error(f"Error loading banana tasks: {e}")
        return {}


class IFTaskPromptManager:
    """
    ComfyUI node to generate task-based prompts for the Gemini node.
    Outputs a simple text string that can be connected to the Gemini prompt input.
    """

    _task_names: list[str] = []

    def __init__(self):
        self.tasks = load_banana_tasks()

    @classmethod
    def INPUT_TYPES(cls):
        tasks = load_banana_tasks()
        cls._task_names = list(tasks.keys()) if tasks else ["No tasks found"]
        
        # Default custom instruction that users can modify
        default_custom = "Analyze the provided images and follow the task instructions above."
        
        return {
            "required": {
                "task": (cls._task_names, {"default": cls._task_names[0] if cls._task_names else ""}),
                "custom_instruction": ("STRING", {
                    "multiline": True, 
                    "default": default_custom,
                    "dynamicPrompts": False
                }),
                "append_mode": (["before", "after", "replace"], {"default": "after"}),
                "raw_prompt_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "separator": ("STRING", {"default": "\n\n", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    def generate_prompt(
        self, 
        task: str, 
        custom_instruction: str = "", 
        append_mode: str = "after",
        separator: str = "\n\n",
        raw_prompt_only: bool = False,
    ) -> Tuple[str]:
        """
        Generate a prompt by combining task system prompt with custom instructions.
        
        Args:
            task: Selected task name from the presets
            custom_instruction: User's custom instruction to append/prepend
            append_mode: How to combine the prompts ("before", "after", "replace")
            separator: Text to use between task prompt and custom instruction
            
        Returns:
            Tuple containing the combined prompt string
        """
        
        # Get the task data
        task_data = self.tasks.get(task)
        if not task_data:
            error_msg = f"Error: Task '{task}' not found in presets."
            logger.error(error_msg)
            return (error_msg,)
        
        # Get the system prompt from the task
        system_prompt = task_data.get("prompt", "")
        
        # If only raw prompt is requested, return it directly
        if raw_prompt_only:
            logger.info(f"Raw prompt only mode active for '{task}'")
            return (system_prompt,)

        # Handle empty custom instruction
        if not custom_instruction or custom_instruction.strip() == "":
            logger.info(f"Using task prompt only for '{task}'")
            return (system_prompt,)
        
        # Combine prompts based on mode
        if append_mode == "replace":
            # Use only custom instruction, ignore task prompt
            combined_prompt = custom_instruction
            logger.info(f"Replaced task prompt with custom instruction")
        elif append_mode == "before":
            # Custom instruction comes before task prompt
            combined_prompt = f"{custom_instruction}{separator}{system_prompt}"
            logger.info(f"Prepended custom instruction to task '{task}'")
        else:  # "after" (default)
            # Task prompt comes first, then custom instruction
            combined_prompt = f"{system_prompt}{separator}{custom_instruction}"
            logger.info(f"Appended custom instruction to task '{task}'")
        
        # Log some details for debugging
        logger.debug(f"Task: {task}")
        logger.debug(f"Append mode: {append_mode}")
        logger.debug(f"Combined prompt length: {len(combined_prompt)} characters")
        
        return (combined_prompt,)


class IFPromptCombiner:
    """
    Simple utility node to combine multiple prompts or add instructions to existing prompts.
    Useful for chaining multiple prompt modifications.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {
                "prompt2": ("STRING", {"multiline": True, "default": ""}),
                "separator": ("STRING", {"default": "\n\n", "multiline": False}),
                "combine_mode": (["append", "prepend", "replace"], {"default": "append"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "combine_prompts"
    CATEGORY = "ImpactFrames💥🎞️/LLM"
    
    def combine_prompts(
        self,
        prompt1: str,
        prompt2: str = "",
        separator: str = "\n\n",
        combine_mode: str = "append"
    ) -> Tuple[str]:
        """
        Combine two prompts with a separator.
        
        Args:
            prompt1: First/main prompt (usually from another node)
            prompt2: Second prompt to combine
            separator: Text to place between prompts
            combine_mode: How to combine ("append", "prepend", "replace")
            
        Returns:
            Tuple containing the combined prompt
        """
        
        # Handle empty prompts
        if not prompt2 or prompt2.strip() == "":
            return (prompt1,)
        
        if not prompt1 or prompt1.strip() == "":
            return (prompt2,)
        
        # Combine based on mode
        if combine_mode == "replace":
            combined = prompt2
        elif combine_mode == "prepend":
            combined = f"{prompt2}{separator}{prompt1}"
        else:  # append
            combined = f"{prompt1}{separator}{prompt2}"
        
        logger.debug(f"Combined prompts using mode '{combine_mode}', total length: {len(combined)}")
        
        return (combined,)