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
import requests
import json

from .env_utils import get_api_key, get_base_url, get_openrouter_api_key, get_effective_api_key
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

# Suppress verbose HTTP logging from the Google API client
# This prevents API keys from being exposed in HTTP request logs and HTML spam
logging.getLogger("httpx").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("httpcore").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("google.genai").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("google.auth").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("urllib3").setLevel(logging.ERROR)  # Add urllib3
logging.getLogger("requests").setLevel(logging.ERROR)  # Add requests


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


def create_appropriate_client(api_key, api_key_source="unknown", force_openrouter=False, force_gemini=False):
    """
    Create the appropriate client (Gemini or OpenRouter) based on configuration
    
    Args:
        api_key: The API key to use for authentication
        api_key_source: Source type of the API key ("openrouter", "gemini", or "unknown")
        force_openrouter: Force OpenRouter base URL even if not configured
        force_gemini: Force Gemini client even if OpenRouter is configured
        
    Returns:
        Configured client (either Gemini client or OpenRouter client)
    """
    # If forcing Gemini, skip OpenRouter checks entirely
    if force_gemini or api_key_source == "gemini":
        # Use standard Gemini client without any base URL modifications
        return create_gemini_client(api_key, api_key_source, force_openrouter=False), "gemini"
    
    base_url = get_base_url()
    
    # Determine if we should use OpenRouter client
    use_openrouter = (
        force_openrouter or 
        api_key_source == "openrouter" or 
        (api_key_source == "unknown" and base_url and "openrouter.ai" in base_url)
    )
    
    if use_openrouter:
        # Use OpenRouter client with OpenAI SDK
        try:
            from .openrouter_client import create_openrouter_client
            openrouter_base_url = base_url or "https://openrouter.ai/api/v1"
            return create_openrouter_client(api_key, openrouter_base_url), "openrouter"
        except ImportError as e:
            logger.error(f"OpenRouter client not available: {e}")
            logger.info("Falling back to Gemini client (may not work with OpenRouter)")
            # Fall through to use Gemini client
        except Exception as e:
            logger.error(f"Failed to create OpenRouter client: {e}")
            logger.info("Falling back to Gemini client")
            # Fall through to use Gemini client
    
    # Use standard Gemini client
    return create_gemini_client(api_key, api_key_source, force_openrouter), "gemini"


def create_gemini_client(api_key, api_key_source="unknown", force_openrouter=False):
    """
    Create Gemini client with configurable base URL and OpenRouter support
    
    Args:
        api_key: The API key to use for authentication
        api_key_source: Source type of the API key ("openrouter", "gemini", or "unknown")
        force_openrouter: Force OpenRouter base URL even if not configured
        
    Returns:
        Configured Gemini client
    """
    from google import genai
    from google.genai import types
    
    # Mask the API key for logging
    masked_key = api_key[:5] + "..." if len(api_key) > 5 else "****"
    
    # If api_key_source is explicitly "gemini", don't use any custom base URL
    if api_key_source == "gemini":
        logger.info(f"Creating standard Gemini client (API key: {masked_key}, source: gemini)")
        return genai.Client(api_key=api_key)
    
    base_url = get_base_url()
    
    # Force OpenRouter URL if using OpenRouter key and no base URL is set
    if force_openrouter and not base_url and api_key_source == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        logger.info("Force-enabling OpenRouter base URL for OpenRouter API key")
    
    if base_url:
        # Use custom base URL if provided
        logger.info(f"Creating Gemini client with custom base URL: {base_url} (API key: {masked_key}, source: {api_key_source})")
        
        # Prepare headers for OpenRouter if using OpenRouter base URL
        headers = {}
        if "openrouter.ai" in base_url:
            # Add OpenRouter-specific headers for better tracking and features
            site_url = os.environ.get("OPENROUTER_SITE_URL", "")
            site_name = os.environ.get("OPENROUTER_SITE_NAME", "ComfyUI-IF_Gemini")
            
            if site_url:
                headers["HTTP-Referer"] = site_url
                logger.debug(f"Added OpenRouter site URL: {site_url}")
            
            if site_name:
                headers["X-Title"] = site_name
                logger.debug(f"Added OpenRouter site name: {site_name}")
            
            if api_key_source == "openrouter":
                logger.info("Using OpenRouter API key with OpenRouter base URL - optimal configuration")
            else:
                logger.warning("Using non-OpenRouter API key with OpenRouter base URL - this may cause issues")
        
        # Create HTTP options with custom headers if needed
        http_options = types.HttpOptions(base_url=base_url)
        if headers:
            # Note: The google.genai client may not support custom headers in HttpOptions
            # In that case, the headers would need to be handled differently
            logger.debug(f"OpenRouter headers prepared: {headers}")
        
        client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )
    else:
        # Use default endpoint
        logger.debug(f"Creating Gemini client with default endpoint (API key: {masked_key}, source: {api_key_source})")
        client = genai.Client(api_key=api_key)
    
    return client


class UniversalClient:
    """
    Universal client wrapper that provides a consistent interface for both Gemini and OpenRouter clients
    """
    
    def __init__(self, client, client_type: str):
        self.client = client
        self.client_type = client_type
        self.logger = logging.getLogger(__name__)
    
    def generate_content(self, contents, model_name: str, generation_config=None, **kwargs):
        """Generate content using either Gemini or OpenRouter client"""
        if self.client_type == "openrouter":
            return self._generate_content_openrouter(contents, model_name, generation_config, **kwargs)
        else:
            return self._generate_content_gemini(contents, model_name, generation_config, **kwargs)
    
    def _generate_content_openrouter(self, contents, model_name: str, generation_config=None, **kwargs):
        """Generate content using OpenRouter client"""
        try:
            # Normalize model name for OpenRouter (expects google/ prefixed IDs)
            def _normalize_model_for_openrouter(model_id: str) -> str:
                if not isinstance(model_id, str):
                    return model_id
                if model_id.startswith("google/"):
                    return model_id
                # Preserve explicit :free suffix if provided
                suffix = ""
                if model_id.endswith(":free"):
                    model_id, suffix = model_id[:-5], ":free"
                mapping = {
                    "gemini-2.5-flash-image-preview": "google/gemini-2.5-flash-image-preview",
                    "gemini-2.5-flash": "google/gemini-2.5-flash",
                    "gemini-2.5-pro": "google/gemini-2.5-pro",
                    "gemini-2.5-flash-002": "google/gemini-2.5-flash-002",
                    "gemini-2.0-flash-exp": "google/gemini-2.0-flash-exp",
                    "gemini-2.0-flash": "google/gemini-2.0-flash",
                    "gemini-2.0-pro": "google/gemini-2.0-pro",
                }
                base = mapping.get(model_id, f"google/{model_id}")
                return base + suffix

            # Extract prompt and images from Gemini-style contents
            prompt = ""
            images = []
            
            if isinstance(contents, list) and len(contents) > 0:
                for content in contents:
                    if isinstance(content, dict) and "parts" in content:
                        for part in content["parts"]:
                            if isinstance(part, dict):
                                if "text" in part:
                                    prompt += part["text"] + " "
                                elif "inline_data" in part:
                                    # Convert inline image data to PIL Image; supports raw bytes or base64
                                    try:
                                        import base64
                                        from io import BytesIO
                                        image_data = part["inline_data"].get("data")
                                        if isinstance(image_data, (bytes, bytearray)):
                                            image_bytes = bytes(image_data)
                                        else:
                                            image_bytes = base64.b64decode(image_data)
                                        image = Image.open(BytesIO(image_bytes))
                                        images.append(image)
                                    except Exception as e:
                                        self.logger.warning(f"Failed to process inline image: {e}")
                            elif hasattr(part, 'text'):
                                prompt += part.text + " "
                    elif isinstance(content, str):
                        prompt += content + " "
                    else:
                        # Assume this could be a PIL Image (from prepare_batch_images)
                        try:
                            if isinstance(content, Image.Image):
                                images.append(content)
                        except Exception:
                            pass
            
            prompt = prompt.strip()
            
            # Extract generation parameters
            max_tokens = None
            temperature = 0.7
            
            if generation_config:
                if hasattr(generation_config, 'max_output_tokens'):
                    max_tokens = generation_config.max_output_tokens
                if hasattr(generation_config, 'temperature'):
                    temperature = generation_config.temperature
            
            # Use OpenRouter client to generate content
            response_text = self.client.generate_content(
                prompt=prompt,
                model=_normalize_model_for_openrouter(model_name),
                images=images if images else None,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Return in Gemini-style format
            class MockResponse:
                def __init__(self, text):
                    self.text = text
                    self.parts = [MockPart(text)]
            
            class MockPart:
                def __init__(self, text):
                    self.text = text
            
            # If the OpenRouter client returned rich dict with images, convert to Gemini-like parts
            if isinstance(response_text, dict):
                text_out = response_text.get("text", "")
                images_bytes = response_text.get("images", []) or []
                class MockImagePart:
                    def __init__(self, data):
                        class Inline:
                            def __init__(self, d):
                                self.mime_type = "image/png"
                                self.data = d
                        self.inline_data = Inline(data)
                        self.text = None
                parts = []
                if text_out:
                    parts.append(MockPart(text_out))
                for b in images_bytes:
                    parts.append(MockImagePart(b))
                class MockResponseWithImages:
                    def __init__(self, p):
                        self.text = text_out
                        self.parts = p
                        self.candidates = [type("Cand", (), {"content": type("Cnt", (), {"parts": p})(), "finish_reason": None})()]
                return MockResponseWithImages(parts)

            return MockResponse(response_text)
            
        except Exception as e:
            self.logger.error(f"OpenRouter content generation failed: {e}")
            raise
    
    def _generate_content_gemini(self, contents, model_name: str, generation_config=None, **kwargs):
        """Generate content using Gemini client"""
        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generation_config,
            **kwargs
        )
    
    def chats(self):
        """Return chat interface"""
        if self.client_type == "openrouter":
            # OpenRouter doesn't have direct chat support through our wrapper yet
            # For now, return None and handle in calling code
            return None
        else:
            return self.client.chats
    
    def models(self):
        """Return models interface"""
        if self.client_type == "openrouter":
            class MockModels:
                def __init__(self, client):
                    self.client = client
                
                def list(self):
                    return self.client.list_models()
            
            return MockModels(self.client)
        else:
            return self.client.models


class IFGeminiAdvanced:
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
            masked_key = self.api_key[:5] + "..." if len(self.api_key) > 5 else "****"
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
                        # Standard Gemini models
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash-002",
                        "gemini-2.5-flash-image-preview",
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-pro",
                        "gemini-2.0-flash",
                        # OpenRouter-specific Gemini models
                        "google/gemini-2.5-flash",
                        "google/gemini-2.5-pro",
                        "google/gemini-2.5-flash-image-preview",
                        "google/gemini-2.5-flash-image-preview:free",
                        "google/gemini-2.0-flash-exp",
                    ],
                    {"default": "gemini-2.5-flash"},
                ),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "sequential_generation": ("BOOLEAN", {"default": False}),
                "batch_count": ("INT", {"default": 4, "min": 1, "max": 20}),
                "aspect_ratio": (
                    ["none", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5"],
                    {"default": "none"},
                ),
                "api_provider": (
                    ["auto", "gemini", "openrouter"], 
                    {"default": "auto"}
                ),
                "external_api_key": ("STRING", {"default": ""}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
                "api_call_delay": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
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
        model_name="gemini-2.5-flash",
        temperature=0.4,
        images=None,
        video=None,
        audio=None,
        seed=0,
        sequential_generation=False,
        batch_count=1,
        aspect_ratio="none",
        api_provider="auto",
        external_api_key="",
        chat_mode=False,
        clear_history=False,
        structured_output=False,
        max_images=6,
        max_output_tokens=8192,
        use_random_seed=False,
        api_call_delay=1.0,
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

        # Enhanced API key resolution with manual provider selection
        api_key = None
        api_key_source = None
        key_type = None
        
        # Clean and validate external API key
        cleaned_external_key = external_api_key.strip() if external_api_key else ""
        
        if cleaned_external_key:
            # External key provided - use provider selector to determine type
            if api_provider == "openrouter":
                key_type = "openrouter"
                logger.info("Using external API key as OpenRouter key (manual selection)")
            elif api_provider == "gemini":
                key_type = "gemini" 
                logger.info("Using external API key as Gemini key (manual selection)")
            else:  # api_provider == "auto"
                # Auto-detect based on base URL
                base_url = get_base_url()
                if base_url and "openrouter.ai" in base_url:
                    key_type = "openrouter"
                    logger.info("Using external API key as OpenRouter key (auto-detected from base URL)")
                else:
                    key_type = "gemini"
                    logger.info("Using external API key as Gemini key (auto-detected)")
            
            api_key = cleaned_external_key
            api_key_source = "external"
            # Save it for future reference
            self.last_external_api_key = cleaned_external_key
        else:
            # No external key, use environment keys with provider selection
            if api_provider == "openrouter":
                # Force OpenRouter
                api_key = get_openrouter_api_key()
                if api_key:
                    key_type = "openrouter"
                    api_key_source = "environment"
                    logger.info("Using OPENROUTER_API_KEY from environment (manual selection)")
                else:
                    logger.warning("OpenRouter provider selected but OPENROUTER_API_KEY not found")
            elif api_provider == "gemini":
                # Force Gemini
                api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if api_key:
                    key_type = "gemini"
                    api_key_source = "environment"
                    logger.info("Using GEMINI_API_KEY from environment (manual selection)")
                else:
                    logger.warning("Gemini provider selected but GEMINI_API_KEY not found")
            else:  # api_provider == "auto"
                # Use automatic resolution
                api_key, key_type = get_effective_api_key()
                api_key_source = "automatic"
                
            if not api_key:
                # Fallback to previously cached external key
                if self.last_external_api_key:
                    api_key = self.last_external_api_key
                    api_key_source = "cached"
                    key_type = "unknown"  # Type unknown for cached keys
                    logger.info("Using previously provided external API key")

        if not api_key:
            if api_provider == "openrouter":
                return (
                    "ERROR: OpenRouter provider selected but OPENROUTER_API_KEY not found. "
                    "Please set OPENROUTER_API_KEY in your environment or provide it in the external_api_key field.",
                    create_placeholder_image(),
                )
            elif api_provider == "gemini":
                return (
                    "ERROR: Gemini provider selected but GEMINI_API_KEY not found. "
                    "Please set GEMINI_API_KEY in your environment or provide it in the external_api_key field.",
                    create_placeholder_image(),
                )
            else:  # auto mode
                base_url = get_base_url()
                if base_url and "openrouter.ai" in base_url:
                    return (
                        "ERROR: OpenRouter base URL detected but no API key found. Please set OPENROUTER_API_KEY "
                        "in your environment or provide it in the external_api_key field.",
                        create_placeholder_image(),
                    )
                else:
                    return (
                        "ERROR: No API key found. Please set GEMINI_API_KEY or OPENROUTER_API_KEY "
                        "in your environment or provide it in the external_api_key field. "
                        "You can also use the api_provider selector to choose a specific provider.",
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
                model_name=model_name,
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
                api_call_delay=api_call_delay,
                api_provider=api_provider,
            )

        # Check for potential compatibility issues with OpenRouter
        if key_type == "openrouter" and api_provider == "openrouter":
            # Issue a warning about OpenRouter compatibility
            logger.warning("IMPORTANT: Google's genai client may not be fully compatible with OpenRouter API.")
            logger.warning("If you encounter issues, consider using OpenAI SDK with OpenRouter instead.")

        # Initialize the API client with the API key
        try:
            # Force the appropriate client based on explicit provider selection
            force_openrouter = (api_provider == "openrouter")
            force_gemini = (api_provider == "gemini")
            raw_client, client_type = create_appropriate_client(api_key, key_type, force_openrouter, force_gemini)
            client = UniversalClient(raw_client, client_type)
            logger.info(f"Using {client_type} client for API calls")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error initializing Gemini client: {error_msg}", exc_info=True)
            
            # Check for OpenRouter compatibility issues and clean up HTML spam
            if key_type == "openrouter" and ("<!DOCTYPE html>" in error_msg or "404 Not Found" in error_msg or len(error_msg) > 1000):
                return (
                    "ERROR: OpenRouter API incompatibility detected during client initialization. "
                    "Google's genai client cannot properly communicate with OpenRouter's OpenAI-compatible API. "
                    "Consider using an OpenAI SDK-based solution for reliable OpenRouter access.",
                    create_placeholder_image(),
                )
            
            # Truncate very long error messages to prevent log spam
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... [truncated]"
            
            # Provide more helpful messages for common errors
            if key_type == "openrouter":
                return (
                    "ERROR: Failed to initialize client with OpenRouter. Google's genai client may not be "
                    "compatible with OpenRouter API. Consider using an OpenAI SDK-based solution instead. "
                    f"Technical error: {error_msg}",
                    create_placeholder_image(),
                )
            elif "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
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
                    if client.client_type == "openrouter":
                        # OpenRouter doesn't support chat sessions directly
                        # Fall back to single message generation
                        logger.info("OpenRouter detected: using single message generation instead of chat session")
                        contents = [{"role": "user", "parts": current_message_parts}]
                        response = client.generate_content(contents, model_name, generation_config)
                    else:
                        # Create chat session with proper history (Gemini)
                        chat_session = client.chats().create(
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
                response = client.generate_content(
                    contents=contents,
                    model_name=model_name,
                    generation_config=generation_config,
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
            error_msg = str(e)
            logger.error(f"Error generating content: {error_msg}")
            
            # Check for OpenRouter compatibility issues and clean up HTML spam
            if key_type == "openrouter" and ("<!DOCTYPE html>" in error_msg or "404 Not Found" in error_msg):
                generated_content = ("ERROR: OpenRouter API incompatibility detected. Google's genai client "
                                   "cannot properly communicate with OpenRouter's OpenAI-compatible API. "
                                   "Consider using an OpenAI SDK-based solution for reliable OpenRouter access.")
            else:
                # Truncate very long error messages to prevent log spam
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... [truncated]"
                generated_content = f"Error: {error_msg}"

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
        api_call_delay=1.0,
        api_provider="auto",
    ):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Import here to avoid ImportError during ComfyUI startup
            from google import genai
            from google.genai import types

            # This determination of `client_type` must happen before the model name is validated.
            # We peek ahead to see what client will be created.
            pre_client_type = "gemini" # Default
            if api_provider == "openrouter":
                pre_client_type = "openrouter"
            elif api_provider == "auto":
                base_url = get_base_url()
                if base_url and "openrouter.ai" in base_url:
                    pre_client_type = "openrouter"

            # For OpenRouter, model names are expected to have a prefix like "google/".
            # For native Gemini, they do not. We adjust if we are using Gemini and the model is not image-capable.
            if pre_client_type == "gemini":
                # List of known official Gemini models that support image generation.
                image_capable_models = [
                    "gemini-2.5-flash-image-preview",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-002"
                ]
                if model_name not in image_capable_models:
                    original_model = model_name
                    model_name = "gemini-2.5-flash-image-preview" # A safe default for Gemini
                    logger.warning(
                        f"Model '{original_model}' may not support image generation with the native Gemini client. "
                        f"Switched to '{model_name}'. If using OpenRouter, ensure you select a model with a 'google/' prefix."
                    )
            # For OpenRouter (`pre_client_type == "openrouter"`), we pass the model name as-is,
            # assuming the user has selected a valid image generation model from the list.


            # Use the API key based on the source specified and provider
            api_key = None
            
            # Determine which API key to look for based on provider
            if api_provider == "openrouter":
                env_key_name = "OPENROUTER_API_KEY"
            elif api_provider == "gemini":
                env_key_name = "GEMINI_API_KEY"
            else:  # auto mode - check both
                base_url = get_base_url()
                if base_url and "openrouter.ai" in base_url:
                    env_key_name = "OPENROUTER_API_KEY"
                else:
                    env_key_name = "GEMINI_API_KEY"
            
            if api_key_source == "external" and external_api_key:
                api_key = external_api_key
                logger.info("Using external API key provided in the node for image generation")
            elif api_key_source == "system" and os.environ.get(env_key_name):
                api_key = os.environ.get(env_key_name)
                logger.info(f"Using {env_key_name} from system environment variable for image generation")
            elif api_key_source == "loaded" and self.api_key:
                api_key = self.api_key
                logger.info("Using API key from previously loaded environment for image generation")
            elif api_key_source == "cached" and self.last_external_api_key:
                api_key = self.last_external_api_key
                logger.info("Using cached external API key for image generation")
            elif external_api_key:  # Fallback to direct external key if source not set
                api_key = external_api_key
                logger.info("Using direct external API key for image generation")
            elif os.environ.get(env_key_name):  # Fallback to system env
                api_key = os.environ.get(env_key_name)
                logger.info(f"Using {env_key_name} from system environment for image generation")
            elif self.api_key:  # Fallback to instance variable
                api_key = self.api_key
                logger.info("Using instance API key for image generation")
            elif self.last_external_api_key:  # Last resort cached key
                api_key = self.last_external_api_key
                logger.info("Using last resort cached API key for image generation")
            
            if not api_key:
                error_msg = f"ERROR: No API key available for image generation. "
                if api_provider == "openrouter":
                    error_msg += "Please set OPENROUTER_API_KEY in your environment or provide it in the external_api_key field."
                elif api_provider == "gemini":
                    error_msg += "Please set GEMINI_API_KEY in your environment or provide it in the external_api_key field."
                else:
                    error_msg += "Please set GEMINI_API_KEY or OPENROUTER_API_KEY in your environment or provide it in the external_api_key field."
                return (error_msg, create_placeholder_image())

            # Create Gemini client
            client_key_type = "unknown"  # Initialize to avoid UnboundLocalError
            try:
                # Respect the api_provider selection for image generation
                if api_provider == "openrouter":
                    client_key_type = "openrouter"
                    force_openrouter = True
                    force_gemini = False
                elif api_provider == "gemini":
                    client_key_type = "gemini"
                    force_openrouter = False
                    force_gemini = True
                else:  # auto mode
                    # Determine key type based on base URL
                    base_url = get_base_url()
                    if base_url and "openrouter.ai" in base_url:
                        client_key_type = "openrouter"
                        force_openrouter = True
                        force_gemini = False
                    else:
                        client_key_type = "gemini"
                        force_openrouter = False
                        force_gemini = False
                    
                raw_client, client_type = create_appropriate_client(api_key, client_key_type, force_openrouter, force_gemini)
                client = UniversalClient(raw_client, client_type)
                logger.info(f"Using {client_type} client for image generation")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error initializing Gemini client for image generation: {error_msg}", exc_info=True)
                
                # Check for OpenRouter compatibility issues and clean up HTML spam
                if client_key_type == "openrouter" and ("<!DOCTYPE html>" in error_msg or "404 Not Found" in error_msg or len(error_msg) > 1000):
                    return (
                        "ERROR: OpenRouter API incompatibility detected. Google's genai client cannot properly communicate with OpenRouter's OpenAI-compatible API. Consider using an OpenAI SDK-based solution for reliable OpenRouter access.",
                        create_placeholder_image(),
                    )
                
                # Truncate very long error messages to prevent log spam
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... [truncated]"
                
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

            all_generated_images_bytes = []
            all_generated_text = []
            status_text = ""

            if client_type == "openrouter":
                if sequential_generation:
                    logger.warning("Sequential image generation is not fully supported with OpenRouter and will behave like standard batch generation.")

                api_url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
                    "X-Title": os.environ.get("OPENROUTER_SITE_NAME", "ComfyUI-IF_Gemini")
                }

                ref_images_b64 = []
                if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                    ref_images_pil = prepare_batch_images(images, max_images, max_size=max(target_width, target_height))
                    for img in ref_images_pil:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        ref_images_b64.append(img_str)
                    logger.info(f"Prepared {len(ref_images_b64)} reference images for OpenRouter")

                aspect_string = f" with dimensions {target_width}x{target_height}" if aspect_ratio != "none" else ""
                final_prompt = f"Generate a detailed, high-quality image{aspect_string} of: {prompt}"

                content_parts = [{"type": "text", "text": final_prompt}]
                for b64_image in ref_images_b64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                    })

                messages = [{"role": "user", "content": content_parts}]
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                }
                if batch_count > 1:
                    payload['n'] = batch_count
                if seed != 0:
                    payload['seed'] = seed
                if temperature is not None:
                    payload['temperature'] = temperature

                log_payload = payload.copy()
                if log_payload.get("messages") and log_payload["messages"][0].get("content"):
                    import copy
                    log_payload = copy.deepcopy(payload)
                    for part in log_payload["messages"][0]["content"]:
                        if part.get("type") == "image_url":
                            part["image_url"]["url"] = "[base64 data omitted]"
                logger.info(f"Sending image generation request to OpenRouter with payload: {json.dumps(log_payload, indent=2)}")

                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    response_json = response.json()

                    if response_json.get("choices"):
                        for choice in response_json["choices"]:
                            message = choice.get("message", {})
                            if "images" in message and message["images"]:
                                for img in message["images"]:
                                    base64_string = img.get("image_url", {}).get("url", "")
                                    if "base64," in base64_string:
                                        base64_string = base64_string.split("base64,")[1]
                                    try:
                                        img_bytes = base64.b64decode(base64_string)
                                        all_generated_images_bytes.append(img_bytes)
                                    except Exception as e:
                                        logger.error(f"Error decoding base64 image from OpenRouter: {e}")
                    
                    status_text = f"Generated {len(all_generated_images_bytes)} images from OpenRouter."

                except requests.exceptions.HTTPError as e:
                    error_body = "Could not read error body."
                    try:
                        error_body = response.text
                    except Exception:
                        pass
                    status_text = f"HTTP error from OpenRouter API: {e}. Body: {error_body}"
                    logger.error(status_text)
                except Exception as e:
                    status_text = f"Error during OpenRouter API call: {e}"
                    logger.error(status_text, exc_info=True)
            else:
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
                
                # Sequential generation mode
                if sequential_generation:
                    logger.info(f"Using sequential generation mode for {batch_count} steps")
                    
                    # Initialize history for sequential generation
                    history = []
                    current_prompt = prompt
                    
                    # Process each step in the sequence
                    for i in range(batch_count):
                        # Add delay before each API call except the first one
                        if i > 0 and api_call_delay > 0:
                            logger.info(f"Waiting for {api_call_delay:.1f} seconds before next API call...")
                            time.sleep(api_call_delay)
                        
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
                            response = client.generate_content(
                                contents=content if i == 0 else history, 
                                model_name=model_name, 
                                generation_config=step_config
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
                                                # Skip accidental HTML payloads
                                                if ("<!DOCTYPE html" in part.text) or ("<html" in part.text.lower()):
                                                    pass
                                                else:
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
                        # Add delay before each API call except the first one
                        if i > 0 and api_call_delay > 0:
                            logger.info(f"Waiting for {api_call_delay:.1f} seconds before next API call...")
                            time.sleep(api_call_delay)
                        
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
                            response = client.generate_content(
                                contents=content, 
                                model_name=model_name, 
                                generation_config=batch_config
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
                                                # Skip accidental HTML payloads
                                                if ("<!DOCTYPE html" in part.text) or ("<html" in part.text.lower()):
                                                    pass
                                                else:
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
            
            # Check for OpenRouter compatibility issues in top-level error handling
            original_error = str(e)
            if ("<!DOCTYPE html>" in original_error or "404 Not Found" in original_error or len(original_error) > 1000):
                # Check if this looks like an OpenRouter issue based on base URL
                base_url = get_base_url()
                if base_url and "openrouter.ai" in base_url:
                    error_msg = ("ERROR: OpenRouter API incompatibility detected during image generation. "
                               "Google's genai client cannot properly communicate with OpenRouter's OpenAI-compatible API. "
                               "Consider using an OpenAI SDK-based solution for reliable OpenRouter access.")
                else:
                    # Truncate very long error messages to prevent log spam
                    if len(original_error) > 500:
                        error_msg = f"Error in image generation: {original_error[:500]}... [truncated]"
            
            return error_msg, create_placeholder_image()

def get_available_models(api_key, api_provider=None):
    """Get available Gemini models for a given API key"""
    try:
        from google import genai
        
        # Determine key type based on explicit provider first, then configuration
        if api_provider == "openrouter":
            key_type = "openrouter"
        elif api_provider == "gemini":
            key_type = "gemini"
        else:
            base_url = get_base_url()
            if base_url and "openrouter.ai" in base_url:
                key_type = "openrouter"
            else:
                key_type = "gemini"
        
        # Initialize client with the provided API key
        raw_client, client_type = create_appropriate_client(api_key, key_type)
        client = UniversalClient(raw_client, client_type)
        
        # List available models
        models_response = client.models().list()
        
        # Filter for Gemini models only
        gemini_models = []
        for model in models_response:
            model_name_lower = model.name.lower()
            if "gemini" in model_name_lower:
                # Extract just the model name from the full path
                model_name = model.name.split('/')[-1]
                # For OpenRouter, keep the full OpenRouter model path format
                if key_type == "openrouter" and "/" in model.name:
                    # OpenRouter model names like "google/gemini-2.5-flash-image-preview:free"
                    if model.name.startswith("google/"):
                        gemini_models.append(model.name)
                else:
                    gemini_models.append(model_name)
        
        # Ensure we always have the default models available
        if key_type == "openrouter":
            # Add OpenRouter-specific Gemini models
            default_models = [
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro", 
                "google/gemini-2.5-flash-image-preview",
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemini-2.0-flash-exp",
                "gemini-2.5-flash",  # Also include standard names as fallback
                "gemini-2.5-pro",
                "gemini-2.5-flash-002",
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro",
                "gemini-2.0-flash"
            ]
        else:
            default_models = [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-002",
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro",
                "gemini-2.0-flash"
            ]
        
        for model in default_models:
            if model not in gemini_models:
                gemini_models.append(model)
        
        return gemini_models
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for OpenRouter HTML response and truncate if needed
        if "<!DOCTYPE html>" in error_msg:
            logger.error("Error retrieving models: Received HTML response instead of API data (OpenRouter incompatibility)")
        else:
            # Truncate long error messages
            if len(error_msg) > 300:
                error_msg = error_msg[:300] + "... [truncated]"
            logger.error(f"Error retrieving models: {error_msg}")
        
        # Return appropriate default models based on configuration or provider
        if api_provider == "openrouter":
            return [
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash-image-preview", 
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-002",
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro",
                "gemini-2.0-flash"
            ]
        base_url = get_base_url()
        if base_url and "openrouter.ai" in base_url:
            return [
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash-image-preview", 
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-002",
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro",
                "gemini-2.0-flash"
            ]
        else:
            return [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-002",
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-exp",
                "gemini-2.0-pro",
                "gemini-2.0-flash"
            ]

def check_gemini_api_key(api_key, api_provider="auto"):
    """Check if a Gemini/OpenRouter API key is valid by attempting to list models"""
    try:
        from google import genai
        
        # Determine key type based on provider selector
        if api_provider == "openrouter":
            key_type = "openrouter"
            service_name = "OpenRouter"
            logger.warning("API key validation with OpenRouter may fail due to genai client compatibility issues")
        elif api_provider == "gemini":
            key_type = "gemini"
            service_name = "Gemini"
        else:  # auto mode
            base_url = get_base_url()
            if base_url and "openrouter.ai" in base_url:
                key_type = "openrouter"
                service_name = "OpenRouter"
                logger.warning("API key validation with OpenRouter may fail due to genai client compatibility issues")
            else:
                key_type = "gemini"
                service_name = "Gemini"
        
        # Initialize client with the provided API key
        force_openrouter = (api_provider == "openrouter")
        force_gemini = (api_provider == "gemini")
        raw_client, client_type = create_appropriate_client(api_key, key_type, force_openrouter, force_gemini)
        
        # For validation, we can try the direct validation method if available
        if client_type == "openrouter" and hasattr(raw_client, 'validate_api_key'):
            return raw_client.validate_api_key()
        
        # Otherwise, use the universal client
        client = UniversalClient(raw_client, client_type)
        
        # Try to list models as a simple API test
        models = client.models().list()
        
        # If we get here, the API key is valid
        return True, f"API key is valid. Successfully connected to {service_name} API."
    except Exception as e:
        error_msg = str(e)
        logger.error(f"API key validation error: {error_msg}")
        
        # Determine service name and key type for error messages
        if api_provider == "openrouter":
            service_name = "OpenRouter"
            key_type = "openrouter"
            additional_info = " Note: Google genai client may not be fully compatible with OpenRouter."
        elif api_provider == "gemini":
            service_name = "Gemini"
            key_type = "gemini"
            additional_info = ""
        else:  # auto mode
            base_url = get_base_url()
            if base_url and "openrouter.ai" in base_url:
                service_name = "OpenRouter"
                key_type = "openrouter"
                additional_info = " Note: Google genai client may not be fully compatible with OpenRouter."
            else:
                service_name = "Gemini"
                key_type = "gemini"
                additional_info = ""
        
        # Check for OpenRouter compatibility issues and clean up HTML spam
        if key_type == "openrouter" and ("<!DOCTYPE html>" in error_msg or "404 Not Found" in error_msg or len(error_msg) > 1000):
            return False, ("OpenRouter API incompatibility detected. Google's genai client "
                         "cannot properly communicate with OpenRouter's OpenAI-compatible API. "
                         "Consider using an OpenAI SDK-based solution for reliable OpenRouter access.")
        
        # Truncate very long error messages to prevent log spam
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... [truncated]"
        
        # Provide more helpful error messages
        if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, f"Invalid API key. Please check your {service_name} API key and try again.{additional_info}"
        elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            return False, f"API quota exceeded. You've reached your usage limit for the {service_name} API.{additional_info}"
        else:
            return False, f"Error validating API key: {error_msg}{additional_info}"
