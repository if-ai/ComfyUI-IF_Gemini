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
import logging
import random
import base64
import time
import uuid
import hashlib

from .env_utils import get_api_key
from .utils import ChatHistory
from .image_utils import (
    create_placeholder_image,
    prepare_batch_images,
    process_images_for_comfy,
)
from .response_utils import prepare_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Create a vivid word-picture representation of this image include elements that characterize the subject, costume, prop elemts, the action, the background, layout and composition elements present on the scene, be sure to mention the style and mood of the scene. Like it would a film director or director of photography"}),
                "operation_mode": (
                    ["analysis", "generate_text", "generate_images"],
                    {"default": "generate_images"},
                ),
                "model_name": (
                    [
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-pro",
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-exp-image-generation",
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
                "batch_count": ("INT", {"default": 4, "min": 1, "max": 20}),
                "aspect_ratio": (
                    ["none", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5"],
                    {"default": "none"},
                ),
                "external_api_key": ("STRING", {"default": ""}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "sequential_generation": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
            },
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
        sequential_generation=False,
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

        # API key priority: 1. external_api_key, 2. system env, 3. previously loaded key
        # Track if we need to regenerate the client
        api_key = None
        api_key_source = None
        
        # Clean and validate external API key
        cleaned_external_key = external_api_key.strip() if external_api_key else ""
        
        if cleaned_external_key:
            api_key = cleaned_external_key
            api_key_source = "external"
            logger.info("Using API key provided in the node")
            # Save it for future reference
            self.last_external_api_key = cleaned_external_key
        elif os.environ.get("GEMINI_API_KEY"):
            api_key = os.environ.get("GEMINI_API_KEY")
            api_key_source = "system"
            logger.info("Using API key from system environment variable")
        elif self.api_key:
            api_key = self.api_key
            api_key_source = "loaded" 
            logger.info("Using API key from previously loaded environment")
        elif self.last_external_api_key:  # Fallback to last provided external key
            api_key = self.last_external_api_key
            api_key_source = "cached"
            logger.info("Using previously provided external API key")

        if not api_key:
            return (
                "ERROR: No API key provided. Please set GEMINI_API_KEY in your environment or"
                " provide it in the external_api_key field.",
                create_placeholder_image(),
            )

        if clear_history:
            self.chat_history.clear()

        # Generate a consistent seed for this operation
        operation_seed = generate_consistent_seed(seed, use_random_seed)

        # Handle image generation mode
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
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
                external_api_key=cleaned_external_key,
                api_key_source=api_key_source,
                sequential_generation=sequential_generation,
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

        # Prepare generation config with all necessary parameters
        generation_config_params = {
            "max_output_tokens": max_output_tokens, 
            "temperature": temperature, 
            "seed": operation_seed,
            "safety_settings": safety_settings
        }
        
        # Add response_mime_type for structured output
        if structured_output:
            generation_config_params["response_mime_type"] = "application/json"
            logger.info("Requesting structured JSON output")
            
        generation_config = types.GenerateContentConfig(**generation_config_params)

        try:
            if chat_mode:
                # Handle chat mode with proper history
                history = self.chat_history.get_messages_for_api()

                # Create appropriate content parts based on input type
                contents = prepare_response(
                    prompt,
                    "image" if images is not None else ("video" if video is not None else ("audio" if audio is not None else "text")),
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
                    current_message_parts = contents[0]["parts"]
                else:
                    # Fallback if we get unexpected format
                    current_message_parts = [{"text": prompt}]
                    logger.warning("Unexpected format from prepare_response, using text prompt only")

                try:
                    # Create chat session with proper history
                    chat_session = client.chats.create(
                        model=model_name,
                        history=history
                    )

                    # Send message to chat and get response
                    response = chat_session.send_message(
                        content=current_message_parts,
                        config=generation_config,
                    )

                    # Process the response
                    if structured_output:
                        try:
                            # Try to parse the response as JSON
                            import json
                            raw_text = response.text
                            parsed_json = json.loads(raw_text)
                            # Pretty print the JSON for better readability
                            response_text = json.dumps(parsed_json, indent=2)
                            logger.info("Successfully parsed structured JSON chat response")
                        except (json.JSONDecodeError, Exception) as e:
                            # Fallback to raw text if JSON parsing fails
                            logger.warning(f"Failed to parse structured JSON output in chat: {str(e)}")
                            response_text = f"Warning: Requested JSON output but received non-JSON response:\n\n{response.text}"
                    else:
                        response_text = response.text

                    # Add to history and format response
                    self.chat_history.add_message("user", prompt)
                    self.chat_history.add_message("assistant", response_text)

                    # Return the chat history
                    generated_content = self.chat_history.get_formatted_history()
                    
                except Exception as chat_error:
                    logger.error(f"Error in chat session: {str(chat_error)}", exc_info=True)
                    if "not supported" in str(chat_error).lower() and "json" in str(chat_error).lower():
                        generated_content = f"Error: This model doesn't support structured JSON output in chat mode. Try a different model or disable structured output."
                    else:
                        generated_content = f"Error in chat session: {str(chat_error)}"
                    # Add error to chat history for transparency
                    self.chat_history.add_message("user", prompt)
                    self.chat_history.add_message("assistant", generated_content)

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

                # Generate content using the model
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

                # Process the response, handling structured output if requested
                if structured_output:
                    try:
                        # Try to parse the response as JSON
                        import json
                        raw_text = response.text
                        parsed_json = json.loads(raw_text)
                        # Pretty print the JSON for better readability
                        generated_content = json.dumps(parsed_json, indent=2)
                        logger.info("Successfully parsed structured JSON response")
                    except (json.JSONDecodeError, Exception) as e:
                        # Fallback to raw text if JSON parsing fails
                        logger.warning(f"Failed to parse structured JSON output: {str(e)}")
                        generated_content = f"Warning: Requested JSON output but received non-JSON response:\n\n{response.text}"
                else:
                    # Standard text response
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
        sequential_generation=False,
    ):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Import here to avoid ImportError during ComfyUI startup
            from google import genai
            from google.genai import types

            # Ensure we're using an image generation capable model
            if "image-generation" not in model_name:
                model_name = "gemini-2.0-flash-exp-image-generation"
                logger.info(f"Changed to image generation model: {model_name}")

            # Use the API key based on the source specified
            api_key = None
            
            if api_key_source == "external" and external_api_key:
                api_key = external_api_key
                logger.info("Using external API key provided in the node for image generation")
            elif api_key_source == "system" and os.environ.get("GEMINI_API_KEY"):
                api_key = os.environ.get("GEMINI_API_KEY")
                logger.info("Using API key from system environment variable for image generation")
            elif api_key_source == "loaded" and self.api_key:
                api_key = self.api_key
                logger.info("Using API key from previously loaded environment for image generation")
            elif api_key_source == "cached" and self.last_external_api_key:
                api_key = self.last_external_api_key
                logger.info("Using cached external API key for image generation")
            elif external_api_key:  # Fallback to direct external key if source not set
                api_key = external_api_key
                logger.info("Using direct external API key for image generation")
            elif os.environ.get("GEMINI_API_KEY"):  # Fallback to system env
                api_key = os.environ.get("GEMINI_API_KEY")
                logger.info("Using system environment API key for image generation")
            elif self.api_key:  # Fallback to instance variable
                api_key = self.api_key
                logger.info("Using instance API key for image generation")
            elif self.last_external_api_key:  # Last resort cached key
                api_key = self.last_external_api_key
                logger.info("Using last resort cached API key for image generation")
            
            if not api_key:
                return (
                    "ERROR: No API key available for image generation. Please set GEMINI_API_KEY in your environment or provide it in the external_api_key field.",
                    create_placeholder_image(),
                )

            # Create Gemini client
            try:
                client = genai.Client(api_key=api_key)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error initializing Gemini client for image generation: {error_msg}", exc_info=True)
                
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

            # Use the same seed as passed from generate_content to ensure consistency
            logger.info(f"Using seed for image generation: {seed}")

            # Define aspect ratio dimensions for Imagen 3
            aspect_ratio_dimensions = {
                "none": (1024, 1024),  # Default square format
                "1:1": (1024, 1024),  # Square
                "16:9": (1408, 768),  # Landscape widescreen
                "9:16": (768, 1408),  # Portrait widescreen
                "4:3": (1280, 896),  # Standard landscape
                "3:4": (896, 1280),  # Standard portrait
                "5:4": (1024, 819),  # Medium landscape format
                "4:5": (819, 1024),  # Medium portrait format
            }

            # Get target dimensions based on aspect ratio
            target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
            logger.info(f"Using resolution {target_width}x{target_height} for aspect ratio {aspect_ratio}")

            # Set up generation config with required fields
            gen_config_args = {
                "temperature": temperature,
                "response_modalities": ["Text", "Image"],  # Critical for image generation
                "seed": seed,  # Always include seed in config
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

            generation_config = types.GenerateContentConfig(**gen_config_args)

            # Process reference images if provided
            ref_images = []
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Convert tensor to list of PIL images - resize to match target dimensions
                ref_images = prepare_batch_images(images, max_images, max_size=max(target_width, target_height))
                logger.info(f"Prepared {len(ref_images)} reference images")

            # Initialize collections for results - accumulate all results before returning
            all_generated_images_bytes = []
            all_generated_text = []
            status_text = ""
            
            # Sequential generation mode
            if sequential_generation:
                logger.info(f"Using sequential generation mode for {batch_count} steps")
                
                # Initialize history for sequential generation
                history = []
                current_prompt = prompt
                
                # Process each step in the sequence
                for i in range(batch_count):
                    try:
                        # Generate a unique seed for each step
                        current_seed = (seed + i) % (2**31 - 1)
                        logger.info(f"Sequential step {i+1}/{batch_count} with seed {current_seed}")
                        
                        # Update config with current seed
                        step_config = types.GenerateContentConfig(
                            temperature=temperature,
                            response_modalities=["Text", "Image"],
                            seed=current_seed,
                            safety_settings=[
                                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                            ],
                        )
                        
                        # Prepare content for this step
                        if i == 0:
                            # First step includes reference images if available
                            if ref_images:
                                # Construct prompt with specific dimensions
                                aspect_string = f" with dimensions {target_width}x{target_height}" if aspect_ratio != "none" else ""
                                content_text = f"Generate a sequence of {batch_count} images{aspect_string}. First image: {prompt}"
                                
                                # Combine text and images
                                content = [content_text] + ref_images
                            else:
                                # Include specific dimensions in the prompt
                                if aspect_ratio != "none":
                                    content_text = f"Generate a sequence of {batch_count} images with dimensions {target_width}x{target_height}. First image: {prompt}"
                                else:
                                    content_text = f"Generate a sequence of {batch_count} images. First image: {prompt}"
                                content = content_text
                                
                            # For history, we'll track the first content differently
                            if isinstance(content, list):
                                parts = []
                                for item in content:
                                    if isinstance(item, str):
                                        parts.append({"text": item})
                                    else:  # Assume PIL image
                                        img_byte_arr = BytesIO()
                                        item.save(img_byte_arr, format='PNG')
                                        img_byte_arr = img_byte_arr.getvalue()
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": "image/png",
                                                "data": img_byte_arr
                                            }
                                        })
                                history.append({"role": "user", "parts": parts})
                            else:
                                history.append({"role": "user", "parts": [{"text": content}]})
                        else:
                            # Subsequent steps - continue from previous output
                            content_text = f"Generate the next image in the sequence. Step {i+1} of {batch_count}: {current_prompt}"
                            content = content_text
                            history.append({"role": "user", "parts": [{"text": content_text}]})
                        
                        # Generate content for this step
                        response = client.models.generate_content(
                            model=model_name, contents=content if i == 0 else history, config=step_config
                        )
                        
                        # Process response
                        step_images_bytes = []
                        step_text = ""
                        finish_reason = None
                        
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = candidate.finish_reason
                                    logger.info(f"Step {i+1} finish reason: {finish_reason}")
                                
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    model_parts = []
                                    for part in candidate.content.parts:
                                        # Extract text
                                        if hasattr(part, 'text') and part.text:
                                            step_text += part.text + "\n"
                                            model_parts.append({"text": part.text})
                                        
                                        # Extract image data
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                image_binary = part.inline_data.data
                                                step_images_bytes.append(image_binary)
                                                model_parts.append({
                                                    "inline_data": {
                                                        "mime_type": part.inline_data.mime_type,
                                                        "data": image_binary
                                                    }
                                                })
                                            except Exception as img_error:
                                                logger.error(f"Error extracting image from response: {str(img_error)}")
                                    
                                    # Add model response to history
                                    history.append({"role": "model", "parts": model_parts})
                        
                        # Update current prompt for next iteration with text from this response
                        if step_text.strip():
                            # Use the text response as context for the next prompt
                            current_prompt = step_text.strip()
                            all_generated_text.append(f"Step {i+1}:\n{current_prompt}")
                        else:
                            all_generated_text.append(f"Step {i+1}: No text generated")
                            # Use a generic continue prompt if no text was generated
                            current_prompt = "Continue the sequence"
                        
                        # Accumulate images from this step (don't process them yet)
                        if step_images_bytes:
                            all_generated_images_bytes.extend(step_images_bytes)
                            status_text += f"Step {i+1} (seed {current_seed}): Generated {len(step_images_bytes)} image(s)\n"
                        else:
                            status_text += f"Step {i+1} (seed {current_seed}): No images generated\n"
                            
                            # If no images were generated in this step, we might want to stop the sequence
                            if finish_reason and "SAFETY" in str(finish_reason).upper():
                                status_text += "Step blocked for safety reasons. Stopping sequence.\n"
                                break
                    
                    except Exception as batch_error:
                        error_msg = f"Error in sequence step {i+1}: {str(batch_error)}"
                        logger.error(error_msg, exc_info=True)
                        status_text += error_msg + "\n"
                        # Continue with next step despite error
            
            # Standard batch generation mode
            else:
                # Handle standard batch generation of separate images
                for i in range(batch_count):
                    try:
                        # Generate a unique seed for each batch based on the operation seed
                        # This ensures consistent but different seeds across batches
                        current_seed = (seed + i) % (2**31 - 1)
                        
                        # Create batch-specific configuration with the unique seed
                        batch_config = types.GenerateContentConfig(
                            temperature=temperature,
                            response_modalities=["Text", "Image"],
                            seed=current_seed,
                            safety_settings=[
                                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                            ],
                        )

                        # Log the seed being used
                        logger.info(f"Generating batch {i+1} with seed {current_seed}")
                        
                        # Prepare content for batch
                        if ref_images:
                            # Construct prompt with specific dimensions
                            aspect_string = f" with dimensions {target_width}x{target_height}" if aspect_ratio != "none" else ""
                            content_text = f"Generate a new image{aspect_string}: {prompt}"
                            
                            # Combine text and images
                            content = [content_text] + ref_images
                        else:
                            # Include specific dimensions in the prompt
                            if aspect_ratio != "none":
                                content_text = f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height} of: {prompt}"
                            else:
                                content_text = f"Generate a detailed, high-quality image of: {prompt}"
                            content = content_text

                        # Generate content
                        response = client.models.generate_content(
                            model=model_name, contents=content, config=batch_config
                        )

                        # Process response to extract generated images and text
                        batch_images_bytes = []
                        response_text = ""
                        finish_reason = None

                        # Check for finish reason which might explain why no images were generated
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                # Check finish reason if available
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = candidate.finish_reason
                                    logger.info(f"Finish reason: {finish_reason}")
                                
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        # Extract text
                                        if hasattr(part, 'text') and part.text:
                                            response_text += part.text + "\n"

                                        # Extract image data
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                image_binary = part.inline_data.data
                                                batch_images_bytes.append(image_binary)
                                            except Exception as img_error:
                                                logger.error(
                                                    f"Error extracting image from response: {str(img_error)}"
                                                )

                        # Accumulate images and text (don't process them yet)
                        if batch_images_bytes:
                            all_generated_images_bytes.extend(batch_images_bytes)
                            status_text += (
                                f"Batch {i+1} (seed {current_seed}): Generated {len(batch_images_bytes)} images\n"
                            )
                            if response_text.strip():
                                all_generated_text.append(f"Batch {i+1}:\n{response_text.strip()}")
                        else:
                            # Include finish reason in status if available
                            finish_info = f" (Reason: {finish_reason})" if finish_reason else ""
                            status_text += f"Batch {i+1} (seed {current_seed}): No images found in response{finish_info}\n"
                            
                            # Add more specific guidance for IMAGE_SAFETY or similar issues
                            if finish_reason and "SAFETY" in str(finish_reason).upper():
                                status_text += "The request was blocked for safety reasons. Try modifying your prompt to avoid content that might trigger safety filters.\n"
                            
                            # Include any text response from the model that might explain the issue
                            if response_text.strip():
                                status_text += f"Model message: {response_text.strip()}\n"
                                all_generated_text.append(f"Batch {i+1} (no image):\n{response_text.strip()}")

                    except Exception as batch_error:
                        status_text += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process all accumulated images into tensors for ComfyUI only after all steps/batches are complete
            if all_generated_images_bytes:
                logger.info(f"Processing {len(all_generated_images_bytes)} accumulated images")
                
                try:
                    # Convert bytes to PIL images
                    pil_images = []
                    for img_bytes in all_generated_images_bytes:
                        try:
                            pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
                            pil_images.append(pil_img)
                        except Exception as img_error:
                            logger.error(f"Error converting image bytes to PIL: {str(img_error)}")
                    
                    if not pil_images:
                        raise ValueError("Failed to convert any image bytes to PIL images")
                    
                    # Ensure all images have the same dimensions
                    first_width, first_height = pil_images[0].size
                    for i in range(1, len(pil_images)):
                        if pil_images[i].size != (first_width, first_height):
                            pil_images[i] = pil_images[i].resize((first_width, first_height), Image.LANCZOS)
                    
                    # Convert PIL images to tensor format
                    tensors = []
                    for pil_img in pil_images:
                        img_array = np.array(pil_img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
                        tensors.append(img_tensor)
                    
                    # Concatenate all image tensors into one batch
                    image_tensors = torch.cat(tensors, dim=0)
                    
                    # Get the actual resolution for reporting
                    height, width = image_tensors.shape[1:3]
                    resolution_info = f"Resolution: {width}x{height}"
                    
                    # Format the result text
                    if sequential_generation:
                        result_text = f"Successfully generated {len(all_generated_images_bytes)} sequential images using {model_name}.\n"
                        result_text += f"Initial prompt: {prompt}\n"
                        result_text += f"Starting seed: {seed}\n"
                        if resolution_info:
                            result_text += f"{resolution_info}\n"
                        
                        # Add text for each step
                        if all_generated_text:
                            result_text += "\n----- Generated Sequence -----\n"
                            result_text += "\n\n".join(all_generated_text)
                    else:
                        result_text = f"Successfully generated {len(all_generated_images_bytes)} images using {model_name}.\n"
                        result_text += f"Prompt: {prompt}\n"
                        result_text += f"Starting seed: {seed}\n"
                        if resolution_info:
                            result_text += f"{resolution_info}\n"
                        
                        # Add text for each batch
                        if all_generated_text:
                            result_text += "\n----- Generated Text -----\n"
                            result_text += "\n\n".join(all_generated_text)
                    
                    # Add status information at the end
                    result_text += f"\n\n----- Generation Status -----\n{status_text}"
                    
                    # Return the final accumulated results
                    return result_text, image_tensors
                    
                except Exception as processing_error:
                    error_msg = f"Error processing accumulated images: {str(processing_error)}"
                    logger.error(error_msg, exc_info=True)
                    return (f"{error_msg}\n\nRaw status:\n{status_text}", create_placeholder_image())
            else:
                # No images were generated successfully
                return (
                    f"No images were generated with {model_name}. Details:\n{status_text}",
                    create_placeholder_image(),
                )

        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, create_placeholder_image()

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
