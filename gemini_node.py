import sys
import os
import warnings

# Suppress warnings related to IMAGE_SAFETY finish reason which is normal
warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

# Add site-packages directory to Python's sys.path
'''
site_packages_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
'''
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import io
import logging
import random
import base64
import time
import uuid
import hashlib
import json
import re

# Import PromptServer for real-time UI updates
try:
    from server import PromptServer
    HAS_PROMPTSERVER = True
except ImportError:
    logger.warning("PromptServer could not be imported. Real-time UI updates will be disabled.")
    HAS_PROMPTSERVER = False
    PromptServer = None

from .env_utils import get_api_key
from .utils import ChatHistory
from .image_utils import (
    create_placeholder_image,
    prepare_batch_images,
    process_images_for_comfy,
    tensor_to_pil,
    resize_image_to_dimensions,
    pil_to_tensor,
)
from .response_utils import prepare_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function for safely sending WebSocket messages
def send_ws_message_safe(event_name, data, node_id=None):
    """Safely send a WebSocket message with error handling."""
    if not HAS_PROMPTSERVER or PromptServer is None:
        return False
    
    try:
        # Ensure node_id is in the data if provided
        if node_id and isinstance(data, dict) and "node_id" not in data:
            data["node_id"] = node_id
            
        PromptServer.instance.send_sync(event_name, data)
        return True
    except Exception as e:
        logger.error(f"Error sending WebSocket message '{event_name}': {e}")
        return False

def generate_consistent_seed(input_seed=0, use_random=False):
    """
    Generate a consistent seed for the Gemini API.
    
    This function uses a more reliable approach to seed generation:
    - If input_seed is non-zero and use_random is False, use the input_seed
    - Otherwise, generate a high-quality random seed based on uuid and time
    
    Returns:
        int: A seed value within the INT32 range (0 to 2^31-1)
    """
    max_int32 = 2**31 - 1
    
    if input_seed != 0 and not use_random:
        # Use the provided seed, but ensure it's within INT32 range
        adjusted_seed = input_seed % max_int32
        logger.info(f"Using provided seed (adjusted to INT32 range): {adjusted_seed}")
        return adjusted_seed
    
    # For random seeds, use a more robust method that won't collide with ComfyUI's seed generation
    # Create a unique identifier by combining:
    # 1. A UUID (universally unique)
    # 2. Current high-precision time
    # 3. ComfyUI's random seed (if we wanted to use it)
    
    unique_id = str(uuid.uuid4())
    current_time = str(time.time_ns())  # Nanosecond precision
    random_component = str(random.randint(0, max_int32))
    
    # Combine and hash all components to get a deterministic but high-quality random value
    combined = unique_id + current_time + random_component
    hash_hex = hashlib.md5(combined.encode()).hexdigest()
    
    # Convert first 8 characters of hash to integer and ensure within INT32 range
    hash_int = int(hash_hex[:8], 16) % max_int32
    
    logger.info(f"Generated random seed: {hash_int}")
    return hash_int


class GeminiNode:
    def __init__(self):
        self.api_key = ""
        self.chat_history = ChatHistory()
        self.last_external_api_key = ""  # Track the last external API key
        self.api_key_source = None  # Track where the API key came from
        
        # First check system environment variables
        system_api_key = os.environ.get("GEMINI_API_KEY", "")
        if system_api_key:
            self.api_key = system_api_key
            self.api_key_source = "system environment variables"
            logger.info("Successfully loaded Gemini API key from system environment")
        else:
            # Next, try to directly check shell configuration files
            home_dir = os.path.expanduser("~")
            shell_config_files = [
                os.path.join(home_dir, ".zshrc"),
                os.path.join(home_dir, ".bashrc"),
                os.path.join(home_dir, ".bash_profile")
            ]
            
            import re
            shell_key = None
            shell_source = None
            
            for config_file in shell_config_files:
                if os.path.exists(config_file):
                    logger.debug(f"Checking {config_file} for API key...")
                    try:
                        with open(config_file, 'r') as f:
                            content = f.read()
                            # Look for export VAR=value or VAR=value patterns
                            patterns = [
                                r'export\s+GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?',
                                r'GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?'
                            ]
                            for pattern in patterns:
                                matches = re.findall(pattern, content)
                                if matches:
                                    shell_key = matches[0]
                                    shell_source = os.path.basename(config_file)
                                    logger.info(f"Found Gemini API key in {shell_source}")
                                    # Also set in environment for future use
                                    os.environ["GEMINI_API_KEY"] = shell_key
                                    break
                    except Exception as e:
                        logger.error(f"Error reading {config_file}: {str(e)}")
                if shell_key:
                    break
                    
            if shell_key:
                self.api_key = shell_key
                self.api_key_source = shell_source
                logger.info(f"Successfully loaded Gemini API key from {shell_source}")
            else:
                # Last resort: check .env files
                env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if env_api_key:
                    self.api_key = env_api_key
                    self.api_key_source = ".env file"
                    logger.info("Successfully loaded Gemini API key from .env file")
                else:
                    logger.warning("No Gemini API key found in any location (system env, shell configs, .env). You'll need to provide it in the node.")
        
        # Log key information (masked for security)
        if self.api_key:
            masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "****"
            logger.info(f"Using Gemini API key ({masked_key}) from {self.api_key_source}")
        
        # Check for Google Generative AI SDK
        self.genai_available = self._check_genai_availability()

    def _check_genai_availability(self):
        """Check if Google Generative AI SDK is available"""
        try:
            # Import just to check availability
            from google import genai

            return True
        except ImportError:
            logger.error(
                "Google Generative AI SDK not installed. Install with: pip install google-generativeai"
            )
            return False

    def _get_active_api_key(self, external_api_key=""):
        """
        Determine the active API key to use based on priority order:
        1. External API key passed to the method (if valid)
        2. System environment variable
        3. Previously stored API key in the instance
        
        Args:
            external_api_key (str): External API key provided in the node UI
            
        Returns:
            tuple: (api_key, api_key_source) where:
                api_key (str): The API key to use
                api_key_source (str): Description of where the key came from
        """
        # Clean and validate external API key
        cleaned_external_key = external_api_key.strip() if external_api_key else ""
        
        # Determine API key to use based on priority
        if cleaned_external_key:
            logger.info("Using API key provided in the node")
            # Save it for future reference
            self.last_external_api_key = cleaned_external_key
            return cleaned_external_key, "external"
        
        # Next check system environment variables
        system_key = os.environ.get("GEMINI_API_KEY", "")
        if system_key:
            logger.info("Using API key from system environment variable")
            return system_key, "system"
        
        # Next try the API key already in the instance
        if self.api_key:
            logger.info(f"Using API key from {self.api_key_source}")
            return self.api_key, self.api_key_source
        
        # Last resort: try previously provided external key
        if self.last_external_api_key:
            logger.info("Using previously provided external API key")
            return self.last_external_api_key, "cached"
        
        # No API key found
        logger.warning("No API key found")
        return "", ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Create a vivid word-picture representation of this image include elements that characterize the subject, costume, prop elemts, the action, the background, layout and composition elements present on the scene, be sure to mention the style and mood of the scene. Like it would a film director or director of photography"}),
                "operation_mode": (
                    ["analysis", "generate_text", "generate_images", "generate_sequence"],
                    {"default": "generate_images"},
                ),
                "model_name": (
                    [
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-pro",
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-exp-image-generation",
                        "gemini-1.5-pro-latest",
                        "gemini-1.5-pro",
                        "gemini-1.5-pro-001",
                        "gemini-1.5-pro-vision-latest",
                    ],
                    {"default": "gemini-2.0-flash-exp"},
                ),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "batch_count": ("INT", {"default": 4, "min": 1, "max": 8}),
                "aspect_ratio": (
                    ["none", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5"],
                    {"default": "none"},
                ),
                "external_api_key": ("STRING", {"default": ""}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate_content"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    def generate_content(
        self,
        prompt,
        operation_mode="analysis",
        chat_mode=False,
        clear_history=False,
        images=None,
        video=None,
        audio=None,
        external_api_key="",
        max_images=6,
        batch_count=1,
        seed=0,
        max_output_tokens=8192,
        temperature=0.4,
        structured_output=False,
        aspect_ratio="none",
        use_random_seed=False,
        model_name="gemini-2.0-flash-exp",
        unique_id=None,
        extra_pnginfo=None,
        prompt_=None,
    ):
        """Generate content using Gemini model with various input types."""

        # Check if Google Generative AI SDK is available
        if not self.genai_available:
            return (
                "ERROR: Google Generative AI SDK not installed. Install with: pip install google-generativeai",
                create_placeholder_image(),
            )

        # Import here to avoid ImportError during ComfyUI startup
        try:
            from google import genai
            from google.genai import types
            logger.info(f"Google Generative AI SDK path: {genai.__file__}")
        except ImportError:
            return ("ERROR: Failed to import Google Generative AI SDK", create_placeholder_image())

        # Get API key using the helper method
        api_key, api_key_source = self._get_active_api_key(external_api_key)
        
        if not api_key:
            return (
                "ERROR: No API key provided. Please set GEMINI_API_KEY in your environment or"
                " provide it in the external_api_key field.",
                create_placeholder_image(),
            )

        # --- Robust Prompt Extraction ---
        prompt_text = ""
        actual_prompt_input = prompt # Use the main 'prompt' input first

        # Check if the main prompt input is a dict with nested structure
        if isinstance(actual_prompt_input, dict):
            # If prompt is a dict, try to extract text from common patterns
            if "text" in actual_prompt_input:
                prompt_text = str(actual_prompt_input["text"])
            elif "prompt" in actual_prompt_input:
                prompt_text = str(actual_prompt_input["prompt"])
            elif "inputs" in actual_prompt_input and isinstance(actual_prompt_input["inputs"], dict):
                # Try to extract from a nested inputs structure
                inputs = actual_prompt_input["inputs"]
                if "prompt" in inputs:
                    prompt_text = str(inputs["prompt"])
                elif "text" in inputs:
                    prompt_text = str(inputs["text"])
                else:
                    # If no standard keys found, convert the whole dict to string
                    prompt_text = str(actual_prompt_input)
            else:
                # If no known structure, convert the whole dict to string
                prompt_text = str(actual_prompt_input)
            logger.info(f"Extracted prompt from dictionary: {prompt_text[:50]}...")
        elif isinstance(actual_prompt_input, (list, tuple)):
            # If prompt is a list or tuple, join elements
            prompt_text = " ".join(str(item) for item in actual_prompt_input)
            logger.info(f"Converted list/tuple prompt to string: {prompt_text[:50]}...")
        else:
            # Otherwise, ensure it's a string
            prompt_text = str(actual_prompt_input)

        # Check if prompt_text is empty or just whitespace, but we have prompt_ (workflow data)
        if not prompt_text.strip() and prompt_ and isinstance(prompt_, dict):
            try:
                logger.debug("Extracted prompt is empty, trying workflow data (prompt_)")
                # Try to find the node data in the workflow
                for node_id, node_data in prompt_.items():
                    if isinstance(node_data, dict) and node_data.get("class_type") == "GeminiNode":
                        if "inputs" in node_data and "prompt" in node_data["inputs"]:
                            prompt_text = str(node_data["inputs"]["prompt"])
                            logger.info(f"Extracted prompt from workflow data: {prompt_text[:50]}...")
                            break
            except Exception as e:
                logger.error(f"Error extracting prompt from workflow data: {e}")

        logger.info(f"Using final prompt text: {prompt_text[:100]}...")
        # --- End of Robust Prompt Extraction ---

        if clear_history:
            self.chat_history.clear()

        # Generate a consistent seed for this operation
        operation_seed = generate_consistent_seed(seed, use_random_seed)

        # Handle different operation modes
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt_text,
                model_name=model_name
                if "image-generation" in model_name
                else "gemini-2.0-flash-exp-image-generation",
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                seed=operation_seed,
                max_images=max_images,
                aspect_ratio=aspect_ratio,
                use_random_seed=use_random_seed,
                external_api_key=api_key,
                api_key_source=api_key_source,
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
                prompt_=prompt_,
            )
        elif operation_mode == "generate_sequence":
            # Call new sequence generation method
            return self.generate_sequence(
                prompt=prompt_text,
                model_name=model_name,
                images=images,
                video=video,
                audio=audio,
                temperature=temperature,
                seed=operation_seed,
                max_output_tokens=max_output_tokens,
                external_api_key=api_key,
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
                prompt_=prompt_,
            )

        # Initialize the API client with the API key
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error initializing Gemini client: {error_msg}", exc_info=True)
            
            # Provide more helpful messages for common errors
            if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return (
                    "ERROR: Invalid Gemini API key. Please check your API key and try again.",
                    create_placeholder_image(),
                )
            elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                return (
                    "ERROR: API quota exceeded. You've reached your usage limit for the Gemini API.",
                    create_placeholder_image(),
                )
                
            return (f"Error initializing Gemini client: {error_msg}", create_placeholder_image())

        # Configure safety settings and generation parameters
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens, 
            temperature=temperature, 
            seed=operation_seed,  # Use our consistent seed
            safety_settings=safety_settings
        )

        try:
            if chat_mode:
                # Handle chat mode with proper history
                history = self.chat_history.get_messages_for_api()

                # Create chat session
                chat_session = client.chats.create(
                    model=model_name,
                    history=history
                )

                # Create appropriate content parts based on input type
                contents = prepare_response(
                    prompt,
                    "image" if images is not None else "text",
                    None,
                    images,
                    video,
                    audio,
                    max_images,
                )
                # Extract content for chat format
                if (
                    isinstance(contents, list)
                    and len(contents) == 1
                    and isinstance(contents[0], dict)
                    and "parts" in contents[0]
                ):
                    contents = contents[0]["parts"]

                # Send message to chat and get response
                response = chat_session.send_message(
                    content=contents,
                    config=generation_config,
                )

                # Add to history and format response
                self.chat_history.add_message("user", prompt)
                self.chat_history.add_message("assistant", response.text)

                # Return the chat history
                generated_content = self.chat_history.get_formatted_history()

            else:
                # Standard non-chat mode - prepare content for each input type
                contents = prepare_response(
                    prompt,
                    "image" if images is not None else "text",
                    None,
                    images,
                    video,
                    audio,
                    max_images,
                )

                # Add structured output instruction if requested
                if structured_output:
                    if (
                        isinstance(contents, list)
                        and len(contents) > 0
                        and isinstance(contents[0], dict)
                        and "parts" in contents[0]
                        and len(contents[0]["parts"]) > 0
                    ):
                        if (
                            isinstance(contents[0]["parts"][0], dict)
                            and "text" in contents[0]["parts"][0]
                        ):
                            contents[0]["parts"][0][
                                "text"
                            ] = f"Please provide the response in a structured format. {contents[0]['parts'][0]['text']}"

                # Generate content using the model
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

                generated_content = response.text

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}", exc_info=True)
            generated_content = f"Error: {str(e)}"

        # For analysis mode, return the text response and an empty placeholder image
        return (generated_content, create_placeholder_image())

    def generate_images(
        self,
        prompt,
        model_name,
        images=None,
        batch_count=1,
        temperature=0.4,
        seed=0,
        max_images=6,
        aspect_ratio="none",
        use_random_seed=False,
        external_api_key="",
        api_key_source=None,
        unique_id=None,
        extra_pnginfo=None,
        prompt_=None,
    ):
        """Generate images using Gemini models with image generation capabilities.
           NOTE: Makes ONE API call. Batch_count > 1 doesn't multiply API calls here.
           It might influence the prompt if we adapt it later, but currently ignored.
           The API itself might return multiple images (up to its limit).
        """
        logger.info(f"Starting generate_images for node {unique_id}")

        if not self.genai_available:
            return ("ERROR: Google Generative AI SDK not installed.", create_placeholder_image())

        # --- Force the Correct Model ---
        # Use the specific model designed for this integrated generation
        forced_model_name = "gemini-2.0-flash-exp-image-generation"
        if model_name != forced_model_name:
             logger.warning(f"Operation mode is generate_images, overriding model to {forced_model_name}")
        model_name = forced_model_name # Override

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ("ERROR: Failed to import Google Generative AI SDK", create_placeholder_image())

        # --- API Key ---
        # Get API key using the helper method 
        api_key, api_key_source = self._get_active_api_key(external_api_key)
        
        if not api_key:
            return (
                "ERROR: No API key available for image generation.",
                create_placeholder_image(),
            )

        # Handle prompt - ensure it's a string before processing
        if isinstance(prompt, dict):
            # If prompt is a dict, try to extract text from it
            if "text" in prompt:
                prompt_text = str(prompt["text"])
            else:
                # If no text key, convert the whole dict to string
                prompt_text = str(prompt)
            logger.info(f"Converted dict prompt to string for image generation: {prompt_text[:50]}...")
        elif isinstance(prompt, (list, tuple)):
            # If prompt is a list or tuple, join elements
            prompt_text = " ".join(str(item) for item in prompt)
            logger.info(f"Converted list/tuple prompt to string for image generation: {prompt_text[:50]}...")
        else:
            # Otherwise, ensure it's a string
            prompt_text = str(prompt)

        # --- Initialize Client ---
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error initializing Gemini client for image generation: {error_msg}", exc_info=True)
            if "invalid api key" in error_msg.lower(): 
                return ("ERROR: Invalid Gemini API key.", create_placeholder_image())
            if "quota" in error_msg.lower(): 
                return ("ERROR: API quota exceeded.", create_placeholder_image())
            return (f"Error initializing Gemini client: {error_msg}", create_placeholder_image())

        # --- Seed for this single call ---
        # Use the consistent seed generated earlier
        current_seed = seed
        logger.info(f"Using seed for image generation call: {current_seed}")

        # --- Aspect Ratio / Dimensions ---
        aspect_ratio_dimensions = {
            "none": (1024, 1024), "1:1": (1024, 1024), "16:9": (1408, 768),
            "9:16": (768, 1408), "4:3": (1280, 896), "3:4": (896, 1280),
            "5:4": (1024, 819), "4:5": (819, 1024),
        }
        target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
        logger.info(f"Target resolution {target_width}x{target_height} for aspect ratio {aspect_ratio}")
        dimension_prompt_part = f" Ensure the image dimensions are {target_width}x{target_height}." if aspect_ratio != "none" else ""

        # --- Generation Config ---
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["Text", "Image"], # MUST include both
            seed=current_seed,
            safety_settings=[ # Keep safety settings relaxed for now
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            # candidate_count is NOT directly applicable here like in text generation for multiple choices.
            # The API might return multiple images based on its internal logic, not this count.
        )

        # --- Prepare Content ---
        content_parts = []
        # Add prompt text, including dimension request
        content_parts.append({"text": prompt_text + dimension_prompt_part})

        # Process and add reference images if provided
        if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            pil_images = []
            # Handle batch dim
            num_ref_images = min(images.shape[0], 6) if images.dim() == 4 else 1
            image_source = images if images.dim() == 4 else images.unsqueeze(0) # Add batch dim if single

            for i in range(num_ref_images):
                try:
                    pil_img = tensor_to_pil(image_source[i])
                    # Resize reference images reasonably
                    pil_img = resize_image_to_dimensions(pil_img, 1024, 1024, crop_to_fit=False)
                    # Convert PIL to bytes
                    buffer = BytesIO()
                    pil_img.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    # Add image part
                    content_parts.append({
                        "inline_data": { "mime_type": "image/png", "data": image_bytes }
                    })
                except Exception as ref_img_e:
                    logger.error(f"Error processing reference image {i}: {ref_img_e}")

        api_content = {"parts": content_parts}

        # --- Make ONE API Call ---
        all_generated_images = []
        response_text_parts = []
        status_text = ""
        finish_reason = "unknown"
        final_text = "No text returned by the model."  # Initialize to default value

        try:
            logger.info(f"Calling Gemini model {model_name} for image generation (Node ID: {unique_id}, Seed: {current_seed})")
            send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": f"🔄 Calling {model_name} (Seed: {current_seed})..."})

            response = client.models.generate_content(
                model=model_name, contents=api_content, config=generation_config
            )

            # Process the response and extract parts
            all_generated_images = []
            response_text_parts = []
            finish_reason = "UNKNOWN"
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                if hasattr(response.candidates[0], "finish_reason"):
                    finish_reason = response.candidates[0].finish_reason
                
                for part_idx, part in enumerate(response.candidates[0].content.parts):
                    if hasattr(part, 'text') and part.text:
                        response_text_parts.append(part.text)
                        send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": part.text, "part_index": f"text_{part_idx}"})

                    elif hasattr(part, 'inline_data') and part.inline_data and 'image' in part.inline_data.mime_type:
                        try:
                            # Extract and add the image to the collection
                            image_bytes = part.inline_data.data
                            all_generated_images.append(image_bytes) # Store raw bytes
                            # Send image part to UI if needed
                            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            send_ws_message_safe("if-gemini-sequence-image", {"node_id": unique_id, "image_b64": img_b64, "mime_type": part.inline_data.mime_type, "part_index": f"img_{part_idx}"})
                        except Exception as img_e:
                            logger.error(f"Error processing image part: {img_e}")
                            send_ws_message_safe("if-gemini-sequence-error", {"node_id": unique_id, "message": f"Error processing image: {img_e}"})
            else:
                 if response.candidates and hasattr(response.candidates[0], "finish_reason"):
                    finish_reason = response.candidates[0].finish_reason
                 logger.warning(f"No valid parts found. Finish reason: {finish_reason}")
                 status_text = f"No content parts received. Reason: {finish_reason}"
                 send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": status_text})

            # Update final_text based on response_text_parts
            final_text = "\n".join(response_text_parts) if response_text_parts else "No text returned by the model."

        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_text = error_msg
            send_ws_message_safe("if-gemini-sequence-error", {"node_id": unique_id, "message": error_msg})


        # Process images if any were generated
        num_images_generated = len(all_generated_images)
        
        if num_images_generated > 0:
            logger.info(f"Generated {num_images_generated} images, processing for ComfyUI...")
            image_tensors_list = []
            
            for i, img_bytes in enumerate(all_generated_images[:max_images]):
                try:
                    # Convert bytes to PIL image and then to tensor
                    pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    # Match aspect ratio if requested
                    if aspect_ratio != "none":
                        target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
                        pil_image = resize_image_to_dimensions(pil_image, target_width, target_height)
                    # Convert to tensor
                    img_tensor = pil_to_tensor(pil_image)
                    image_tensors_list.append(img_tensor)
                except Exception as tensor_e:
                    logger.error(f"Error processing image {i} to tensor: {tensor_e}")
            
            # If we have processed images, batch them; otherwise return placeholder
            if not image_tensors_list:
                 result_text = f"Generated {num_images_generated} image(s) but failed to process them.\nReason: {finish_reason}\n{status_text}\nModel Text: {final_text}"
                 send_ws_message_safe("if-gemini-sequence-complete", {"node_id": unique_id, "message": "Processing failed."})
                 return (result_text, create_placeholder_image())

            # Process multiple images
            if len(image_tensors_list) > 1:
                # Ensure all tensors have the same dimensions
                height, width = image_tensors_list[0].shape[1:3]
                for i in range(1, len(image_tensors_list)):
                    if image_tensors_list[i].shape[1:3] != (height, width):
                        # Resize to match the first image's dimensions
                        pil_img = tensor_to_pil(image_tensors_list[i])
                        pil_img = pil_img.resize((width, height), Image.LANCZOS)
                        image_tensors_list[i] = pil_to_tensor(pil_img)
                
                # Now batch all tensors
                final_image_batch = torch.cat(image_tensors_list, dim=0)
            else:
                # Single image case
                final_image_batch = image_tensors_list[0]
                
            num_images_returned = len(image_tensors_list)
            logger.info(f"Returning {num_images_returned} processed images")
            
            # Check for requested image count from prompt
            requested_count = None
            prompt_lower = prompt_text.lower()
            count_match = re.search(r'generate\s+(\d+)\s+images', prompt_lower)
            if count_match:
                try:
                    requested_count = int(count_match.group(1))
                except ValueError:
                    requested_count = None
            
            # Add explanation about batch_count vs. API generation
            api_note = ""
            if batch_count > 1:
                api_note = f"\nNote: batch_count={batch_count} was provided, but the Gemini API makes a single API call that may return multiple images based on its internal logic."
            
            if requested_count and requested_count > num_images_generated and num_images_generated > 0:
                api_note += f"\nNote: You requested {requested_count} images in the prompt, but the API returned {num_images_generated}. The Gemini API determines the appropriate number of images based on the context and content requirements."
                
            # Format final text output
            result_text = (
                f"Generated {num_images_generated} images, returned {num_images_returned}.\n"
                f"Reason: {finish_reason}\n"
                f"Model Text: {final_text}"
                f"{api_note}"
            )
            
            # Create more informative completion message
            completion_message = f"Complete: {num_images_returned} images."
            if requested_count and requested_count > num_images_generated:
                completion_message = f"Complete: {num_images_returned} images (API returned fewer images than the {requested_count} requested)."
                
            send_ws_message_safe("if-gemini-sequence-complete", {
                "node_id": unique_id, 
                "message": completion_message,
                "requested_count": requested_count if requested_count else None,
                "api_count": num_images_generated,
                "processed_count": num_images_returned
            })
            
            return (result_text, final_image_batch)

        else:
            # No images generated
            result_text = f"No images generated by {model_name}. Reason: {finish_reason}\n{status_text}\nModel Text: {final_text}"
            send_ws_message_safe("if-gemini-sequence-complete", {"node_id": unique_id, "message": "Complete: No images generated."})
            return (result_text, create_placeholder_image())

    def generate_sequence(
        self,
        prompt,
        model_name,
        images=None,
        video=None,
        audio=None,
        temperature=0.4,
        seed=0,
        max_output_tokens=8192,
        max_images=4,
        aspect_ratio="none",
        external_api_key="",
        unique_id=None,
        extra_pnginfo=None,
        prompt_=None,
    ):
        logger.info(f"Starting generate_sequence for node {unique_id}")
        # Check for SDK availability
        if not self.genai_available:
            return ("ERROR: Google Generative AI SDK not installed.", create_placeholder_image())

        # Send initialization message to UI if available
        send_ws_message_safe("if-gemini-sequence-init", {"node_id": unique_id})
        
        if not unique_id:
            logger.warning("Node unique_id is missing, UI updates may be incomplete")
            
        # Force the correct model for generation regardless of user selection
        enforced_model = "gemini-2.0-flash-exp-image-generation"
        if model_name != enforced_model:
            logger.warning(f"Sequence generation requires {enforced_model}. Overriding selected model {model_name}.")
            model_name = enforced_model
        
        # Get API key with proper logic
        api_key, api_key_source = self._get_active_api_key(external_api_key)
        if not api_key:
            return ("ERROR: No API key found. Please provide a valid Gemini API key.", create_placeholder_image())
            
        try:
            # Import SDK
            from google import genai
            from google.genai import types
            
            # Initialize client
            try:
                client = genai.Client(api_key=api_key)
            except Exception as client_e:
                error_msg = f"Failed to initialize API client: {str(client_e)}"
                logger.error(error_msg)
                send_ws_message_safe("if-gemini-sequence-error", {"node_id": unique_id, "message": error_msg})
                return (error_msg, create_placeholder_image())
                
            # Prepare content for API call
            aspect_ratio_dimensions = {
                "none": (1024, 1024),
                "1:1": (1024, 1024),
                "16:9": (1408, 768),
                "9:16": (768, 1408),
                "4:3": (1280, 896),
                "3:4": (896, 1280),
                "5:4": (1024, 819),
                "4:5": (819, 1024),
            }
            target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
            
            # Add aspect ratio request to prompt if specified
            dimension_prompt = ""
            if aspect_ratio != "none":
                dimension_prompt = f" Please create images with aspect ratio {aspect_ratio} ({target_width}x{target_height})."
            
            # Handle prompt - ensure it's a string before processing
            if isinstance(prompt, dict):
                # If prompt is a dict, try to extract text from it
                if "text" in prompt:
                    prompt_text = str(prompt["text"])
                else:
                    # If no text key, convert the whole dict to string
                    prompt_text = str(prompt)
                logger.info(f"Converted dict prompt to string: {prompt_text[:50]}...")
            elif isinstance(prompt, (list, tuple)):
                # If prompt is a list or tuple, join elements
                prompt_text = " ".join(str(item) for item in prompt)
                logger.info(f"Converted list/tuple prompt to string: {prompt_text[:50]}...")
            else:
                # Otherwise, ensure it's a string
                prompt_text = str(prompt)
                
            # Enhance prompt for sequence generation
            prompt_lower = prompt_text.lower()
            if "generate" in prompt_lower and "images" in prompt_lower:
                # Try to extract number if user asked for specific count
                count_match = re.search(r'generate\s+(\d+)\s+images', prompt_lower)
                count_str = count_match.group(1) if count_match else "multiple"
                enhanced_prompt = f"Create a sequence of text descriptions and corresponding images for the following request, aiming for around {count_str} steps if possible: {prompt_text}{dimension_prompt}"
                logger.info(f"Enhanced sequence prompt: {enhanced_prompt[:100]}...")
                final_prompt = enhanced_prompt
            else:
                final_prompt = f"{prompt_text}{dimension_prompt}"
                
            # Prepare content parts
            content_parts = [{"text": final_prompt}]
            
            # Add reference images if provided
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Handle single image
                if images.dim() == 3:
                    images = images.unsqueeze(0)  # Add batch dimension [1, H, W, C]
                
                # Limit number of reference images to process
                num_ref_images = min(images.shape[0], 6)  # Process up to 6 reference images
                logger.info(f"Adding {num_ref_images} reference images to request")
                
                for i in range(num_ref_images):
                    try:
                        # Convert tensor to PIL image
                        pil_img = tensor_to_pil(images[i])
                        # Resize image for API (max 4MB)
                        pil_img = resize_image_to_dimensions(pil_img, 1024, 1024, crop_to_fit=False)
                        # Convert to bytes
                        img_buf = BytesIO()
                        pil_img.save(img_buf, format="PNG")
                        img_bytes = img_buf.getvalue()
                        # Add to content parts
                        content_parts.append({"inline_data": {"mime_type": "image/png", "data": img_bytes}})
                    except Exception as img_e:
                        logger.error(f"Error processing reference image {i}: {str(img_e)}")
            
            # Configure generation settings
            safety_settings = [
                {"category": cat, "threshold": "BLOCK_NONE"} 
                for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                           "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
            ]
            
            generation_config = types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image'],  # This is key for mixed text+image output
                safety_settings=safety_settings
            )
            
            # Make API call
            logger.info(f"Calling Gemini model {model_name} for sequence generation (Node ID: {unique_id}, Seed: {seed})")
            send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": f"🔄 Calling {model_name} (Seed: {seed})..."})
            
            response = client.models.generate_content(
                model=model_name,
                contents={"parts": content_parts},
                config=generation_config
            )
            
            # Process response
            all_text_parts = []
            all_image_tensors = []
            finish_reason = "UNKNOWN"
            
            if response.candidates and response.candidates[0].content:
                if hasattr(response.candidates[0], "finish_reason"):
                    finish_reason = response.candidates[0].finish_reason.name
                    logger.info(f"Response finish reason: {finish_reason}")
                
                # Process each part in the response
                for part_idx, part in enumerate(response.candidates[0].content.parts):
                    if hasattr(part, 'text') and part.text:
                        logger.debug(f"Node {unique_id}: Received text part")
                        all_text_parts.append(part.text)
                        send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": part.text, "part_index": f"text_{part_idx}"})
                        
                    elif hasattr(part, 'inline_data') and part.inline_data and 'image' in part.inline_data.mime_type:
                        try:
                            logger.debug(f"Node {unique_id}: Received image part {part_idx}")
                            # Process the image
                            image_bytes = part.inline_data.data
                            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            send_ws_message_safe("if-gemini-sequence-image", {"node_id": unique_id, "image_b64": img_b64, "mime_type": part.inline_data.mime_type, "part_index": f"img_{part_idx}"})
                            
                            # Convert to tensor for final output batch
                            if len(all_image_tensors) < max_images:  # Only process up to max_images
                                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                                # Apply aspect ratio if specified
                                if aspect_ratio != "none":
                                    pil_image = resize_image_to_dimensions(pil_image, target_width, target_height)
                                # Convert to tensor
                                img_tensor = pil_to_tensor(pil_image)
                                all_image_tensors.append(img_tensor)
                            else:
                                logger.warning(f"Node {unique_id}: Skipping image {part_idx} as max_images limit reached")
                        except Exception as img_e:
                            logger.error(f"Node {unique_id}: Error processing image part: {str(img_e)}", exc_info=True)
                            send_ws_message_safe("if-gemini-sequence-error", {"node_id": unique_id, "message": f"Error processing image: {str(img_e)}", "part_index": f"img_{part_idx}"})
                    else:
                        logger.warning(f"Node {unique_id}: Received unsupported part type or empty part")
            else:
                logger.warning(f"No valid parts found in response. Finish reason: {finish_reason}")
                status_text = f"Generation finished but no content received. (Reason: {finish_reason})"
                send_ws_message_safe("if-gemini-sequence-text", {"node_id": unique_id, "text": status_text})
                all_text_parts.append(status_text)
            
            # Prepare final outputs
            # Create batched tensor from all images
            if all_image_tensors:
                try:
                    # Ensure all tensors are the same size for batching
                    target_h, target_w = all_image_tensors[0].shape[1:3]
                    resized_tensors = []
                    
                    for tensor in all_image_tensors:
                        if tensor.shape[1] != target_h or tensor.shape[2] != target_w:
                            logger.debug(f"Resizing image from {tensor.shape[1]}x{tensor.shape[2]} to {target_h}x{target_w}")
                            pil_img = tensor_to_pil(tensor.squeeze(0))
                            resized = resize_image_to_dimensions(pil_img, target_w, target_h)
                            resized_tensor = pil_to_tensor(resized)
                            resized_tensors.append(resized_tensor)
                        else:
                            resized_tensors.append(tensor)
                    
                    final_image_batch = torch.cat(resized_tensors, dim=0)
                except Exception as batch_e:
                    logger.error(f"Error creating image batch: {str(batch_e)}")
                    final_image_batch = create_placeholder_image()
            else:
                # No images in response
                final_image_batch = create_placeholder_image()
            
            # Create final text output with improved information
            final_text = "\n\n".join(all_text_parts) if all_text_parts else "No text generated."
            
            # Check if user requested specific number of images
            requested_count = None
            count_match = re.search(r'generate\s+(\d+)\s+images', prompt_lower)
            if count_match:
                try:
                    requested_count = int(count_match.group(1))
                except ValueError:
                    requested_count = None
            
            # Add information about image generation limitations if applicable
            num_images_generated = len(all_image_tensors)
            api_note = ""
            if requested_count and requested_count > num_images_generated and num_images_generated > 0:
                api_note = (f"\n\nNote: You requested {requested_count} images, but the API returned {num_images_generated}. "
                            f"The Gemini API often returns fewer images than requested as it determines "
                            f"the appropriate number based on the context and content requirements.")
            
            # Create result text with enhanced information
            result_text = f"Sequence generation with {model_name}\n"
            result_text += f"Finish reason: {finish_reason}\n"
            result_text += f"Generated {len(all_text_parts)} text parts and {num_images_generated} images.\n"
            if api_note:
                result_text += api_note
            result_text += f"\n\n{final_text}"
            
            # Send final completion message
            completion_message = f"✅ Sequence generation complete: {len(all_text_parts)} text parts, {num_images_generated} images. Reason: {finish_reason}"
            if requested_count and requested_count > num_images_generated:
                completion_message += f" (Note: API returned fewer images than the {requested_count} requested)"
                
            send_ws_message_safe("if-gemini-sequence-complete", {
                "node_id": unique_id, 
                "message": completion_message, 
                "text_count": len(all_text_parts), 
                "image_count": num_images_generated
            })
            
            logger.info(f"Node {unique_id}: Sequence finished. Returning {len(all_text_parts)} text parts and {num_images_generated} images.")
            return (result_text, final_image_batch)
            
        except Exception as e:
            error_msg = f"Error during sequence generation: {str(e)}"
            logger.error(f"Node {unique_id}: {error_msg}", exc_info=True)
            send_ws_message_safe("if-gemini-sequence-error", {"node_id": unique_id, "message": error_msg})
            return (error_msg, create_placeholder_image())

def get_available_models(api_key):
    """Get available Gemini models for a given API key"""
    try:
        from google import genai
        
        # Initialize client with the provided API key
        client = genai.Client(api_key=api_key)
        
        # List available models
        models_response = client.models.list()
        
        # Filter for Gemini models only
        gemini_models = []
        for model in models_response:
            if "gemini" in model.name.lower():
                # Extract just the model name from the full path
                model_name = model.name.split('/')[-1]
                gemini_models.append(model_name)
        
        # Ensure we always have the default models available
        default_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation"
        ]
        
        for model in default_models:
            if model not in gemini_models:
                gemini_models.append(model)
        
        return gemini_models
        
    except Exception as e:
        logger.error(f"Error retrieving models: {str(e)}")
        # Return default models on error
        return [
            "gemini-2.0-flash-exp",
            "gemini-2.0-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation"
        ]

def check_gemini_api_key(api_key):
    """Check if a Gemini API key is valid by attempting to list models"""
    try:
        from google import genai
        
        # Initialize client with the provided API key
        client = genai.Client(api_key=api_key)
        
        # Try to list models as a simple API test
        models = client.models.list()
        
        # If we get here, the API key is valid
        return True, "API key is valid. Successfully connected to Gemini API."
    except Exception as e:
        error_msg = str(e)
        logger.error(f"API key validation error: {error_msg}")
        
        # Provide more helpful error messages
        if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, "Invalid API key. Please check your API key and try again."
        elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            return False, "API quota exceeded. You've reached your usage limit for the Gemini API."
        else:
            return False, f"Error validating API key: {error_msg}"

# Ensure pil_to_tensor function is available
def pil_to_tensor(pil_image):
    """Convert PIL image to tensor with shape [1, H, W, 3]"""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    # Ensure the array has 3 dimensions [H, W, C] before adding batch
    if img_array.ndim == 2:  # Handle grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension [1, H, W, 3]
    return img_tensor
