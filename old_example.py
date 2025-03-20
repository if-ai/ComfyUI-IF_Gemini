import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import logging

from .env_utils import get_api_key
from .utils import ChatHistory
from .image_utils import create_placeholder_image, tensor_to_image, resize_image, process_images_for_comfy
# Note: process_audio is used indirectly through prepare_response
from .response_utils import prepare_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiNode:
    def __init__(self):
        self.api_key = ""
        self.temp_dir = os.path.join(tempfile.gettempdir(), "comfyui_gemini")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Setup chat history
        self.chat_history = ChatHistory()
        
        # Try to load API key from environment
        self.api_key = get_api_key("GEMINI_API_KEY", "Gemini")
        
        # Check for Google Generative AI SDK
        self.genai_available = self._check_genai_availability()

    def _check_genai_availability(self):
        """Check if Google Generative AI SDK is available"""
        try:
            # Import just to check availability
            from google import genai
            return True
        except ImportError:
            print("Google Generative AI SDK not installed. Install with: pip install google-generativeai")
            return False
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail"}),
                "input_type": (["text", "image"], {"default": "image"}),
                "operation_mode": (["analysis", "generate_images"], {"default": "analysis"}),
                "model_version": (["gemini-2.0-flash-exp", "gemini-2.0-pro", "gemini-2.0-flash", 
                                   "gemini-2.0-flash-exp-image-generation"], 
                                  {"default": "gemini-2.0-flash-exp"}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("IMAGE",), 
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 8}),
                "aspect_ratio": (["none", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5"], {"default": "none"}),
                "external_api_key": ("STRING", {"default": ""}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate_content"
    CATEGORY = "ImpactFrames💥🎞️/LLM"
    
    def generate_content(self, prompt, input_type="image", model_version="gemini-2.0-flash-exp", 
                      operation_mode="analysis", chat_mode=False, clear_history=False,
                      images=None, video=None, audio=None, 
                      external_api_key="", max_images=6, batch_count=1, seed=0,
                      max_output_tokens=8192, temperature=0.4, structured_output=False,
                      aspect_ratio="none"):
        """Generate content using Gemini model with various input types."""
        
        # Check if Google Generative AI SDK is available
        if not self.genai_available:
            return ("ERROR: Google Generative AI SDK not installed. Install with: pip install google-generativeai", 
                    create_placeholder_image())
        
        # Import here to avoid ImportError during ComfyUI startup
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ("ERROR: Failed to import Google Generative AI SDK", create_placeholder_image())
        
        # Use external API key if provided, otherwise use environment
        if external_api_key.strip():
            self.api_key = external_api_key
        elif not self.api_key:
            self.api_key = get_api_key("GEMINI_API_KEY", "Gemini")

        if not self.api_key:
            return ("ERROR: No API key provided. Please set GEMINI_API_KEY in your environment or provide it in the external_api_key field.", 
                    create_placeholder_image())

        if clear_history:
            self.chat_history.clear()

        # Automatically detect input type based on what's provided (used for handling different media types)
        if video is not None and isinstance(video, torch.Tensor) and video.nelement() > 0:
            actual_input_type = "video"
        elif audio is not None:
            actual_input_type = "audio"
        elif images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            actual_input_type = "image"
        else:
            actual_input_type = input_type

        # Handle image generation mode
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
                model_version=model_version if "image-generation" in model_version else "gemini-2.0-flash-exp-image-generation",
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                seed=seed,
                max_images=max_images,
                aspect_ratio=aspect_ratio
            )

        # Create Gemini client
        client = genai.Client(api_key=self.api_key)
        
        # For analysis mode (text response)
        safety_settings = [
            {"category": "harassment", "threshold": "ALLOW"},
            {"category": "hate_speech", "threshold": "ALLOW"},
            {"category": "sexually_explicit", "threshold": "ALLOW"},
            {"category": "dangerous_content", "threshold": "ALLOW"},
            {"category": "civic", "threshold": "ALLOW"}
        ]

        generation_config = types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        try:
            if chat_mode:
                # Handle chat mode
                # Get existing history in Gemini API format
                history = self.chat_history.get_messages_for_api()
                
                # Create chat session and send message
                chat_session = client.models.start_chat(
                    model=model_version,
                    history=self.chat_history.get_messages_for_api()
                )
                
                # Create content based on the actual input type
                content = self._create_chat_content(prompt, actual_input_type, images, video, audio, max_images)
                
                # Send message to chat
                response = chat_session.send_message(
                    content=content,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Add to history and format response
                if isinstance(content, list):
                    history_content = prompt
                else:
                    history_content = content
                    
                self.chat_history.add_message("user", history_content)
                self.chat_history.add_message("assistant", response.text)
                
                # Return the chat history
                generated_content = self.chat_history.get_formatted_history()
            else:
                # Non-chat mode
                parts = []

                # Use the prepare_response utility function to handle all input types
                parts = prepare_response(prompt, actual_input_type, None, images, video, audio, max_images)
                
                # Add structured output instruction if requested
                if structured_output and isinstance(parts, list) and len(parts) > 0:
                    if isinstance(parts[0], str):
                        parts[0] = f"Please provide the response in a structured format. {parts[0]}"
                    elif isinstance(parts[0], dict) and "parts" in parts[0] and len(parts[0]["parts"]) > 0:
                        if isinstance(parts[0]["parts"][0], dict) and "text" in parts[0]["parts"][0]:
                            parts[0]["parts"][0]["text"] = f"Please provide the response in a structured format. {parts[0]['parts'][0]['text']}"
                
                # Get the model instance (this is the correct way to use the Gemini API)
                model = client.get_model(model_version)
                
                # Generate content using the model
                response = model.generate_content(
                    parts,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                generated_content = response.text

        except Exception as e:
            import traceback
            print(f"Error generating content: {str(e)}")
            print(traceback.format_exc())
            generated_content = f"Error: {str(e)}"
    
        # For analysis mode, return the text response and an empty placeholder image
        return (generated_content, create_placeholder_image())

    # The _determine_input_type method has been moved inline for simplicity

    # Helper methods have been removed in favor of using prepare_response from response_utils

    def _create_chat_content(self, prompt, input_type, images, video, audio, max_images):
        """Create appropriate content for chat based on input type"""
        # Use the prepare_response utility for content preparation
        content = prepare_response(prompt, input_type, None, images, video, audio, max_images)
        
        # The prepare_response function returns a list with a single dictionary entry for structured content
        # For chat mode, we need to extract the actual content
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and "parts" in content[0]:
            return content[0]["parts"]
        
        return content

    def generate_images(self, prompt, model_version, images=None, batch_count=1, 
                        temperature=0.4, seed=0, max_images=6, aspect_ratio="none"):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Import here to avoid ImportError during ComfyUI startup
            from google import genai
            from google.genai import types
            
            # Create Gemini client
            client = genai.Client(api_key=self.api_key)
            
            # Set up generation config
            generation_config_args = {
                "temperature": temperature,
                "response_modalities": ['Text', 'Image']  # Critical for image generation
            }
            
            # Add seed if provided
            if seed != 0:
                generation_config_args["seed"] = seed
                
            generation_config = types.GenerateContentConfig(**generation_config_args)
            
            # Process reference images if provided
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Convert tensor to list of PIL images
                input_images = []
                if len(images.shape) == 4:  # [batch, H, W, C]
                    num_images = min(images.shape[0], max_images)
                    for i in range(num_images):
                        pil_image = tensor_to_image(images[i])
                        pil_image = resize_image(pil_image, 1024)
                        input_images.append(pil_image)
                else:  # Single image tensor [H, W, C]
                    pil_image = tensor_to_image(images)
                    pil_image = resize_image(pil_image, 1024)
                    input_images.append(pil_image)
                
                # Construct prompt with aspect ratio
                aspect_string = f" with aspect ratio {aspect_ratio}" if aspect_ratio != "none" else ""
                content_text = f"Generate a new image in the style of these reference images{aspect_string}: {prompt}"
                content = [content_text] + input_images
            else:
                # Text-only prompt
                aspect_string = f" with aspect ratio {aspect_ratio}" if aspect_ratio != "none" else ""
                content_text = f"Generate a detailed, high-quality image{aspect_string} of: {prompt}"
                content = content_text
            
            # Track generated images and status
            all_generated_images = []
            status_text = ""
            
            # Generate images for each batch
            for i in range(batch_count):
                try:
                    # Update seed for each batch if provided
                    if seed != 0:
                        current_seed = seed + i
                        batch_config = types.GenerateContentConfig(
                            temperature=temperature,
                            response_modalities=['Text', 'Image'],
                            seed=current_seed
                        )
                    else:
                        batch_config = generation_config
                    
                    # Generate content
                    response = model.generate_content(
                        content,
                        generation_config=batch_config
                    )
                    
                    # Extract generated images and text
                    batch_images = []
                    response_text = ""
                    
                    # Process response parts
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        # Extract text
                                        if hasattr(part, 'text') and part.text:
                                            response_text += part.text + "\n"
                                        
                                        # Extract image data
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                if hasattr(part.inline_data, 'data'):
                                                    batch_images.append(part.inline_data.data)
                                            except Exception as img_error:
                                                print(f"Error extracting image from response: {str(img_error)}")
                    
                    if batch_images:
                        all_generated_images.extend(batch_images)
                        status_text += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        status_text += f"Batch {i+1}: No images found in response\n"
                
                except Exception as batch_error:
                    status_text += f"Batch {i+1} error: {str(batch_error)}\n"
            
            # Process generated images into tensors using the utility function
            if all_generated_images:
                # Create a data structure that process_images_for_comfy can handle
                image_data = {
                    "data": [
                        {"b64_json": img_binary}
                        for img_binary in all_generated_images
                    ]
                }
                
                # Use the utility function to convert to ComfyUI-compatible tensors
                try:
                    image_tensors, mask_tensors = process_images_for_comfy(
                        image_data,
                        response_key="data",
                        field_name="b64_json"
                    )
                    
                    result_text = f"Successfully generated {len(all_generated_images)} images using {model_version}.\n"
                    result_text += f"Prompt: {prompt}\n"
                    if aspect_ratio != "none":
                        result_text += f"Aspect ratio: {aspect_ratio}\n"
                    
                    return result_text, image_tensors
                except Exception as e:
                    error_msg = f"Error processing generated images: {str(e)}"
                    print(error_msg)
                    return error_msg, create_placeholder_image()
            
            # No images were generated successfully
            return f"No images were generated with {model_version}. Details:\n{status_text}", create_placeholder_image()
            
        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            print(error_msg)
            return error_msg, create_placeholder_image()