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

# Import PromptServer for real-time UI updates
try:
    from server import PromptServer
except ImportError:
    # Fallback for environments where server might not be directly available
    logging.warning("PromptServer could not be imported. Real-time UI updates will be disabled.")
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

        # Handle different operation modes
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
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
                prompt_=prompt_,
            )
        elif operation_mode == "generate_sequence":
            # Call new sequence generation method
            return self.generate_sequence(
                prompt=prompt,
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

            # Prepare content for the API
            content = None

            # Process reference images if provided
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Convert tensor to list of PIL images - resize to match target dimensions
                pil_images = prepare_batch_images(images, max_images, max_size=max(target_width, target_height))

                if len(pil_images) > 0:
                    # Construct prompt with specific dimensions
                    aspect_string = (
                        f" with dimensions {target_width}x{target_height}"
                        if aspect_ratio != "none"
                        else ""
                    )
                    content_text = (
                        f"Generate a new image in the style of these reference images{aspect_string}: {prompt}"
                    )

                    # Combine text and images
                    content = [content_text] + pil_images
                else:
                    logger.warning("No valid images found in input tensor")

            # Use text-only prompt if no images or processing failed
            if content is None:
                # Include specific dimensions in the prompt
                if aspect_ratio != "none":
                    content_text = (
                        f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height}"
                        f" of: {prompt}"
                    )
                else:
                    content_text = f"Generate a detailed, high-quality image of: {prompt}"
                content = content_text

            # Track generated images
            all_generated_images = []
            status_text = ""

            # Generate images - handle batch generation with unique seeds
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

                    # Generate content
                    response = client.models.generate_content(
                        model=model_name, contents=content, config=batch_config
                    )

                    # Process response to extract generated images and text
                    batch_images = []
                    response_text = ""
                    finish_reason = None

                    # Check for finish reason which might explain why no images were generated
                    if hasattr(response, "candidates") and response.candidates:
                        for candidate in response.candidates:
                            # Check finish reason if available
                            if hasattr(candidate, "finish_reason"):
                                finish_reason = candidate.finish_reason
                                logger.info(f"Finish reason: {finish_reason}")
                            
                            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                                for part in candidate.content.parts:
                                    # Extract text
                                    if hasattr(part, "text") and part.text:
                                        response_text += part.text + "\n"

                                    # Extract image data
                                    if hasattr(part, "inline_data") and part.inline_data:
                                        try:
                                            image_binary = part.inline_data.data
                                            batch_images.append(image_binary)
                                        except Exception as img_error:
                                            logger.error(
                                                f"Error extracting image from response: {str(img_error)}"
                                            )

                    if batch_images:
                        all_generated_images.extend(batch_images)
                        status_text += (
                            f"Batch {i+1} (seed {current_seed}): Generated {len(batch_images)} images\n"
                        )
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

                except Exception as batch_error:
                    status_text += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process generated images into tensors for ComfyUI
            if all_generated_images:
                # Create a data structure for process_images_for_comfy
                image_data = {
                    "data": [
                        {"b64_json": base64.b64encode(img).decode("utf-8")}
                        for img in all_generated_images
                    ]
                }

                # Use the utility function to convert images
                image_tensors, mask_tensors = process_images_for_comfy(
                    image_data, response_key="data", field_name="b64_json"
                )

                # Get the actual resolution of the first image for information
                if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() >= 3:
                    height, width = image_tensors.shape[1:3]
                    resolution_info = f"Resolution: {width}x{height}"
                else:
                    resolution_info = ""

                result_text = (
                    f"Successfully generated {len(all_generated_images)} images using"
                    f" {model_name}.\n"
                )
                result_text += f"Prompt: {prompt}\n"
                result_text += f"Starting seed: {seed}\n"
                if resolution_info:
                    result_text += f"{resolution_info}\n"

                return result_text, image_tensors

            # No images were generated successfully
            return (
                f"No images were generated with {model_name}. Details:\n{status_text}",
                create_placeholder_image(),
            )

        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, create_placeholder_image()

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
        external_api_key="",
        unique_id=None,
        extra_pnginfo=None,
        prompt_=None,
    ):
        """Generates an interleaved sequence of text and images using Gemini."""
        logger.info(f"Starting generate_sequence for node {unique_id}")

        if not self.genai_available:
            return ("ERROR: Google Generative AI SDK not installed.", create_placeholder_image())
        
        # Check for real-time UI update capability
        if not PromptServer:
            logger.warning("PromptServer not available, cannot send real-time sequence updates.")
        if not unique_id:
            logger.warning("Node unique_id not provided, cannot send real-time sequence updates.")

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ("ERROR: Failed to import Google Generative AI SDK", create_placeholder_image())

        # --- API Key ---
        # We receive the already determined API key via external_api_key argument
        api_key = external_api_key
        if not api_key:
            return ("ERROR: API key missing for sequence generation.", create_placeholder_image())

        # --- Initialize Client ---
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error initializing Gemini client for sequence: {error_msg}", exc_info=True)
            # More specific errors
            if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return ("ERROR: Invalid Gemini API key.", create_placeholder_image())
            if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                return ("ERROR: API quota exceeded.", create_placeholder_image())
            return (f"Error initializing Gemini client: {error_msg}", create_placeholder_image())

        # --- Send Init Message to UI ---
        if PromptServer and unique_id:
            try:
                PromptServer.instance.send_sync("if-gemini-sequence-init", {
                    "node_id": unique_id,
                    "message": "Starting sequence generation...",
                })
            except Exception as ws_e:
                logger.warning(f"Failed to send initialization WebSocket message: {ws_e}")

        # --- Extract actual prompt text ---
        # Ensure we have a clean text-only prompt by removing any workflow data
        actual_prompt = prompt
        if isinstance(prompt, dict) and "inputs" in prompt:
            # This is likely workflow data, extract just the prompt text
            logger.warning("Received prompt as workflow data dictionary, extracting text content")
            if "prompt" in prompt["inputs"] and isinstance(prompt["inputs"]["prompt"], str):
                actual_prompt = prompt["inputs"]["prompt"]
            elif "text" in prompt["inputs"]:
                if isinstance(prompt["inputs"]["text"], list) and len(prompt["inputs"]["text"]) > 0:
                    actual_prompt = " ".join(str(item) for item in prompt["inputs"]["text"])
                elif isinstance(prompt["inputs"]["text"], str):
                    actual_prompt = prompt["inputs"]["text"]
        
        # Also handle the case where it might be wrapped in a list
        elif isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], str):
                actual_prompt = " ".join(str(item) for item in prompt)
            elif isinstance(prompt[0], dict) and "inputs" in prompt[0]:
                # Complex case of list of workflow data
                extracted_parts = []
                for item in prompt:
                    if isinstance(item, dict) and "inputs" in item:
                        if "prompt" in item["inputs"] and isinstance(item["inputs"]["prompt"], str):
                            extracted_parts.append(item["inputs"]["prompt"])
                        elif "text" in item["inputs"]:
                            if isinstance(item["inputs"]["text"], list):
                                extracted_parts.extend(str(t) for t in item["inputs"]["text"])
                            else:
                                extracted_parts.append(str(item["inputs"]["text"]))
                if extracted_parts:
                    actual_prompt = " ".join(extracted_parts)
        
        # If we somehow still have a dict/complex object, convert to string representation
        if not isinstance(actual_prompt, str):
            logger.warning(f"Non-string prompt converted to string: {type(actual_prompt)}")
            actual_prompt = str(actual_prompt)
        
        logger.info(f"Using extracted prompt: {actual_prompt[:100]}...")

        # --- Prepare Contents for API ---
        try:
            # Create a simple content object that will work with the API
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Convert tensor images to PIL images
                pil_images = []
                
                # Handle batch dimension
                if len(images.shape) == 4:  # [batch, H, W, C]
                    batch_size = min(images.shape[0], 6)  # Limit to 6 images
                    for i in range(batch_size):
                        pil_img = tensor_to_pil(images[i])
                        pil_img = resize_image_to_dimensions(pil_img, min(pil_img.width, 1024), min(pil_img.height, 1024))
                        pil_images.append(pil_img)
                else:  # Single image [H, W, C]
                    pil_img = tensor_to_pil(images)
                    pil_img = resize_image_to_dimensions(pil_img, min(pil_img.width, 1024), min(pil_img.height, 1024))
                    pil_images.append(pil_img)
                
                # Check if we have valid images to use
                if pil_images:
                    # For multimodal content (text + images), we need to use the parts format
                    # This is the most reliable format that works with the API
                    parts = []
                    
                    # Add the text part first
                    parts.append({"text": actual_prompt})
                    
                    # Add image parts
                    for pil_img in pil_images:
                        # Convert PIL to bytes
                        buffer = BytesIO()
                        pil_img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        
                        # Add image part using the correct format for the API
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_bytes
                            }
                        })
                    
                    # Simple API-compatible content format
                    content = {"parts": parts}
                else:
                    # Fallback to text-only if no valid images
                    content = actual_prompt
            else:
                # Text-only content (simplest case)
                content = actual_prompt
            
            logger.debug(f"Prepared content for API type: {type(content)}")
        except Exception as e:
            logger.error(f"Error preparing content for sequence API call: {str(e)}", exc_info=True)
            if PromptServer and unique_id:
                PromptServer.instance.send_sync("if-gemini-sequence-error", {
                    "node_id": unique_id,
                    "message": f"Error preparing content: {str(e)}",
                })
            return (f"Error preparing content: {str(e)}", create_placeholder_image())

        # --- Configure Generation ---
        # Crucially set response_modalities for multimodal output
        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            response_modalities=['Text', 'Image'],  # Enable multimodal output
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )

        # --- Call Gemini API ---
        all_text_parts = []
        all_image_tensors = []
        
        try:
            logger.info(f"Calling Gemini model {model_name} for sequence generation (Node ID: {unique_id})")
            
            # Send message to UI about API call starting
            if PromptServer and unique_id:
                PromptServer.instance.send_sync("if-gemini-sequence-text", {
                    "node_id": unique_id,
                    "text": f"🔄 Calling Gemini API with model {model_name}...",
                })
            
            # Log the actual content being sent for debugging
            if isinstance(content, dict) and "parts" in content:
                part_count = len(content["parts"])
                part_types = []
                for part in content["parts"]:
                    if "text" in part:
                        part_types.append("text")
                    elif "inline_data" in part:
                        part_types.append(part["inline_data"].get("mime_type", "unknown"))
                logger.info(f"Sending content with {part_count} parts: {part_types}")
            else:
                logger.info(f"Sending simple text content of length {len(str(content))}")
            
            # Make the API call
            response = client.models.generate_content(
                model=model_name,
                contents=content,
                config=generation_config,
            )
            logger.debug(f"Received API response for node {unique_id}")

            # --- Process Response Parts and Send Updates ---
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part_idx, part in enumerate(response.candidates[0].content.parts):
                    # Process text parts
                    if hasattr(part, 'text') and part.text:
                        logger.debug(f"Node {unique_id}: Received text part")
                        all_text_parts.append(part.text)
                        # Send text update via WebSocket
                        if PromptServer and unique_id:
                            try:
                                PromptServer.instance.send_sync("if-gemini-sequence-text", {
                                    "node_id": unique_id, 
                                    "text": part.text,
                                    "part_index": part_idx
                                })
                            except Exception as ws_e:
                                logger.warning(f"Node {unique_id}: Failed to send text WebSocket update: {ws_e}")

                    # Process image parts
                    elif hasattr(part, 'inline_data') and part.inline_data and 'image' in part.inline_data.mime_type:
                        logger.debug(f"Node {unique_id}: Received image part (mime: {part.inline_data.mime_type})")
                        try:
                            # Get the binary image data
                            image_bytes = part.inline_data.data
                            img_b64 = base64.b64encode(image_bytes).decode('utf-8')

                            # Send image update via WebSocket for display
                            if PromptServer and unique_id:
                                try:
                                    PromptServer.instance.send_sync("if-gemini-sequence-image", {
                                        "node_id": unique_id, 
                                        "image_b64": img_b64, 
                                        "mime_type": part.inline_data.mime_type,
                                        "part_index": part_idx
                                    })
                                except Exception as ws_e:
                                    logger.warning(f"Node {unique_id}: Failed to send image WebSocket update: {ws_e}")

                            # Convert to tensor for final output batch
                            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                            img_tensor = pil_to_tensor(pil_image)  # Should return [1, H, W, 3]
                            all_image_tensors.append(img_tensor)

                        except Exception as img_e:
                            logger.error(f"Node {unique_id}: Error processing image part: {str(img_e)}", exc_info=True)
                            # Send error message via WebSocket
                            if PromptServer and unique_id:
                                PromptServer.instance.send_sync("if-gemini-sequence-error", {
                                    "node_id": unique_id, 
                                    "message": f"Error processing image: {str(img_e)}",
                                    "part_index": part_idx
                                })

                    else:
                        logger.warning(f"Node {unique_id}: Received unsupported part type or empty part")
            else:
                # Handle cases where the response might be blocked or empty
                finish_reason = "unknown"
                if response.candidates and hasattr(response.candidates[0], "finish_reason"):
                    finish_reason = response.candidates[0].finish_reason
                     
                logger.warning(f"Node {unique_id}: No valid parts found in Gemini response. Finish reason: {finish_reason}")
                final_status = f"Generation complete but no content received. (Reason: {finish_reason})"
                
                # Send status update
                if PromptServer and unique_id:
                    PromptServer.instance.send_sync("if-gemini-sequence-text", {
                        "node_id": unique_id, 
                        "text": final_status
                    })
                all_text_parts.append(final_status)

            # --- Prepare Final Node Outputs ---
            # Concatenate all text parts for text output
            final_text = "\n".join(all_text_parts) if all_text_parts else "No text generated in sequence."

            # Create final image batch
            final_image_batch = None
            if all_image_tensors:
                if len(all_image_tensors) > 1:
                    # Get dimensions from first image
                    target_h = all_image_tensors[0].shape[1]
                    target_w = all_image_tensors[0].shape[2]
                    
                    # Resize all images to match first image dimensions
                    resized_tensors = []
                    for tensor in all_image_tensors:
                        if tensor.shape[1] != target_h or tensor.shape[2] != target_w:
                            logger.debug(f"Resizing image from {tensor.shape[1]}x{tensor.shape[2]} to {target_h}x{target_w}")
                            pil_img = tensor_to_pil(tensor.squeeze(0))
                            resized_pil = resize_image_to_dimensions(pil_img, target_w, target_h)
                            resized_tensor = pil_to_tensor(resized_pil)
                            resized_tensors.append(resized_tensor)
                        else:
                            resized_tensors.append(tensor)
                    
                    # Concatenate all tensors into a batch
                    final_image_batch = torch.cat(resized_tensors, dim=0)
                else:
                    # Just one image, use it directly
                    final_image_batch = all_image_tensors[0]
            else:
                # No images generated, return placeholder
                final_image_batch = create_placeholder_image()

            # Send final completion message via WebSocket
            if PromptServer and unique_id:
                completion_message = f"✅ Sequence generation complete: {len(all_text_parts)} text parts and {len(all_image_tensors)} images."
                PromptServer.instance.send_sync("if-gemini-sequence-complete", {
                    "node_id": unique_id, 
                    "message": completion_message,
                    "text_count": len(all_text_parts),
                    "image_count": len(all_image_tensors)
                })

            logger.info(f"Node {unique_id}: Sequence generation finished. Returning {len(all_text_parts)} text parts and {len(all_image_tensors)} images.")
            return (final_text, final_image_batch)

        except Exception as e:
            error_msg = f"Error during sequence generation: {str(e)}"
            logger.error(f"Node {unique_id}: {error_msg}", exc_info=True)
            
            # Send error update via WebSocket
            if PromptServer and unique_id:
                PromptServer.instance.send_sync("if-gemini-sequence-error", {
                    "node_id": unique_id, 
                    "message": error_msg
                })
            
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
