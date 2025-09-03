"""
OpenRouter client using OpenAI SDK for compatibility with OpenRouter's OpenAI-compatible API
"""
import logging
import os
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import base64
from io import BytesIO
import re
import requests

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    OpenRouter client that uses OpenAI SDK for compatibility
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client for OpenRouter"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"OpenRouter client initialized with base URL: {self.base_url}")
        except ImportError:
            logger.error("OpenAI SDK not available. Install with: pip install openai>=1.0.0")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise
    
    def _encode_image(self, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _format_messages(self, prompt: str, images: Optional[List[Image.Image]] = None) -> List[Dict[str, Any]]:
        """Format messages for OpenAI-compatible API"""
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add images if provided
        if images:
            for image in images:
                encoded_image = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
        
        return [{"role": "user", "content": content}]

    def _format_responses_input(self, prompt: str, images: Optional[List[Image.Image]] = None) -> List[Dict[str, Any]]:
        """Format input for OpenAI Responses API with optional images"""
        content: List[Dict[str, Any]] = []
        # Text
        content.append({
            "type": "input_text",
            "text": prompt
        })
        # Images
        if images:
            for image in images:
                encoded_image = self._encode_image(image)
                content.append({
                    "type": "input_image",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })
        return [{"role": "user", "content": content}]

    def _try_fetch_image(self, url: str) -> Optional[bytes]:
        """Download URL and return bytes only if it resolves to an image.
        Validates using content-type and PIL verification."""
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200 or not r.content:
                return None
            content_type = (r.headers.get("content-type") or "").lower()
            if content_type.startswith("image/"):
                return r.content
            # As a fallback, try to verify bytes with PIL
            try:
                img = Image.open(BytesIO(r.content))
                img.verify()
                return r.content
            except Exception:
                return None
        except Exception:
            return None

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Best-effort conversion of OpenAI SDK objects to dict for parsing."""
        try:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
        except Exception:
            pass
        try:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
        except Exception:
            pass
        try:
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            import json
            if hasattr(obj, "model_dump_json"):
                return json.loads(obj.model_dump_json())
            if hasattr(obj, "json"):
                return json.loads(obj.json())
        except Exception:
            pass
        # Last resort: string-coerce and attempt URL scrape only
        return {"_string": str(obj)}

    def _gather_images_and_text(self, data: Any) -> Dict[str, Any]:
        """Recursively search for images (base64 or URLs) and text in arbitrary data."""
        images: List[bytes] = []
        texts: List[str] = []

        def walk(node: Any):
            if node is None:
                return
            if isinstance(node, dict):
                # Base64 fields
                for key in ("image_base64", "b64_json"):
                    if key in node and node[key]:
                        try:
                            images.append(base64.b64decode(node[key]))
                        except Exception:
                            pass
                # URL fields
                url = None
                if "url" in node and isinstance(node["url"], str) and node["url"].startswith("http"):
                    url = node["url"]
                elif "image_url" in node and node["image_url"]:
                    if isinstance(node["image_url"], dict):
                        url = node["image_url"].get("url")
                    else:
                        url = getattr(node["image_url"], "url", None)
                if url:
                    try:
                        r = requests.get(url, timeout=15)
                        if r.status_code == 200 and r.content:
                            images.append(r.content)
                    except Exception:
                        pass
                # Data URL
                if "url" in node and isinstance(node["url"], str) and node["url"].startswith("data:image"):
                    try:
                        header, b64 = node["url"].split(",", 1)
                        images.append(base64.b64decode(b64))
                    except Exception:
                        pass
                # Text fields
                for tk in ("text", "output_text", "content"):
                    v = node.get(tk)
                    if isinstance(v, str):
                        texts.append(v)
                # Recurse
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, str):
                texts.append(node)
                for u in re.findall(r"https?://\S+", node):
                    try:
                        r = requests.get(u, timeout=15)
                        if r.status_code == 200 and r.content:
                            images.append(r.content)
                    except Exception:
                        pass

        walk(data)
        return {"text": "\n".join([t for t in texts if t]), "images": images}
    
    def generate_content(self, 
                        prompt: str, 
                        model: str = "google/gemini-2.0-flash-thinking-exp",
                        images: Optional[List[Image.Image]] = None,
                        max_tokens: Optional[int] = None,
                        temperature: float = 0.7,
                        **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Generate text content using OpenRouter
        
        Args:
            prompt: Text prompt
            model: Model name (OpenRouter format)
            images: Optional list of PIL Images
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prefer Responses API for multimodal/image-capable models
            use_responses = bool(images) or ("image-preview" in model)
            if use_responses and hasattr(self.client, "responses"):
                input_payload = self._format_responses_input(prompt, images)
                params: Dict[str, Any] = {
                    "model": model,
                    "input": input_payload,
                    "temperature": temperature,
                }
                if max_tokens:
                    params["max_output_tokens"] = max_tokens
                logger.info(f"Making OpenRouter Responses request to model: {model}")
                response = self.client.responses.create(**params)

                # First, try generic parser across possible shapes
                try:
                    generic = self._gather_images_and_text(self._to_dict(response))
                    if generic.get("images") or generic.get("text"):
                        return generic
                except Exception:
                    pass

                # Parse output for text and images
                text_out = ""
                images_out: List[bytes] = []

                # Try multiple shapes defensively
                try:
                    output = getattr(response, "output", None) or getattr(response, "data", None) or []
                    for item in output:
                        content_list = getattr(item, "content", None) or []
                        for piece in content_list:
                            # Text
                            if hasattr(piece, "text") and piece.text:
                                text_out += piece.text
                            # Base64 image
                            if hasattr(piece, "image_base64") and piece.image_base64:
                                images_out.append(base64.b64decode(piece.image_base64))
                            elif hasattr(piece, "b64_json") and piece.b64_json:
                                images_out.append(base64.b64decode(piece.b64_json))
                            # URL image
                            image_url = None
                            if hasattr(piece, "image_url") and piece.image_url:
                                # piece.image_url may be dict-like or obj with url
                                if isinstance(piece.image_url, dict):
                                    image_url = piece.image_url.get("url")
                                else:
                                    image_url = getattr(piece.image_url, "url", None)
                            if image_url:
                                img_bytes = self._try_fetch_image(image_url)
                                if img_bytes:
                                    images_out.append(img_bytes)
                            # Some providers emit inline base64 as data:image/... in 'url'
                            if hasattr(piece, "url") and isinstance(piece.url, str) and piece.url.startswith("data:image"):
                                try:
                                    header, b64 = piece.url.split(",", 1)
                                    images_out.append(base64.b64decode(b64))
                                except Exception:
                                    pass
                except Exception as parse_err:
                    logger.warning(f"Fallback parsing Responses output: {parse_err}")

                # Fallbacks: output_text, choices
                if not text_out:
                    text_out = getattr(response, "output_text", None) or ""
                if not images_out and hasattr(response, "choices") and response.choices:
                    try:
                        # Some providers embed images in choices[].message.content[*].image_base64
                        for ch in response.choices:
                            msg = getattr(ch, "message", None)
                            if not msg:
                                continue
                            content = getattr(msg, "content", None) or []
                            for part in content:
                                if getattr(part, "type", None) == "output_image":
                                    b64 = getattr(part, "image_base64", None) or getattr(part, "b64_json", None)
                                    if b64:
                                        images_out.append(base64.b64decode(b64))
                                elif getattr(part, "type", None) == "text" and getattr(part, "text", None):
                                    text_out += part.text
                    except Exception:
                        pass

                # If provider returned an URL in text, try to fetch it as an image
                if not images_out and isinstance(text_out, str):
                    urls = re.findall(r"https?://\S+", text_out)
                    for u in urls:
                        img_bytes = self._try_fetch_image(u)
                        if img_bytes:
                            images_out.append(img_bytes)

                return {"text": text_out, "images": images_out}

            # Fallback to Chat Completions (text-only)
            messages = self._format_messages(prompt, images)
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            for key, value in kwargs.items():
                if key not in request_params and value is not None:
                    request_params[key] = value
            logger.info(f"Making OpenRouter Chat request to model: {model}")
            response = self.client.chat.completions.create(**request_params)
            # Try generic parser first
            try:
                generic = self._gather_images_and_text(self._to_dict(response))
                if generic.get("images") or generic.get("text"):
                    return generic
            except Exception:
                pass
            if response.choices and response.choices[0].message:
                msg = response.choices[0].message
                content = getattr(msg, "content", None)
                # If content is a string, try URL fetch
                if isinstance(content, str):
                    if content.startswith("http"):
                        img_bytes = self._try_fetch_image(content)
                        if img_bytes:
                            return {"text": content, "images": [img_bytes]}
                    return content
                # If content is a list of parts
                images_out: List[bytes] = []
                text_out = ""
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            ptype = part.get("type")
                            if ptype == "output_image":
                                url = (part.get("image_url") or {}).get("url")
                                if url:
                                    img_bytes = self._try_fetch_image(url)
                                    if img_bytes:
                                        images_out.append(img_bytes)
                            elif ptype == "text" and part.get("text"):
                                text_out += part.get("text")
                        else:
                            # object-like
                            ptype = getattr(part, "type", None)
                            if ptype == "output_image":
                                iu = getattr(part, "image_url", None)
                                url = iu.get("url") if isinstance(iu, dict) else getattr(iu, "url", None)
                                if url:
                                    img_bytes = self._try_fetch_image(url)
                                    if img_bytes:
                                        images_out.append(img_bytes)
                            elif ptype == "text" and getattr(part, "text", None):
                                text_out += getattr(part, "text")
                if images_out or text_out:
                    return {"text": text_out, "images": images_out}
                return ""
            logger.warning("Empty response from OpenRouter chat")
            return ""

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def list_models(self) -> List:
        """
        List available models from OpenRouter
        
        Returns:
            List of model objects with .name attribute
        """
        try:
            models = self.client.models.list()
            
            # Create mock model objects that match Gemini API structure
            class MockModel:
                def __init__(self, model_id):
                    self.name = model_id
                    self.id = model_id
            
            model_objects = []
            for model in models.data:
                model_objects.append(MockModel(model.id))
            
            logger.info(f"Found {len(model_objects)} models on OpenRouter")
            return model_objects
            
        except Exception as e:
            logger.error(f"Failed to list OpenRouter models: {e}")
            return []
    
    def validate_api_key(self) -> tuple[bool, str]:
        """
        Validate the OpenRouter API key
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Try to list models as a simple validation
            models = self.client.models.list()
            if models.data:
                return True, f"OpenRouter API key is valid. Found {len(models.data)} models."
            else:
                return False, "OpenRouter API key validation failed: No models returned."
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenRouter API key validation error: {error_msg}")
            
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "OpenRouter API key is invalid or unauthorized."
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return False, "OpenRouter API key lacks required permissions."
            else:
                return False, f"OpenRouter API key validation failed: {error_msg}"


def create_openrouter_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenRouterClient:
    """
    Create and return an OpenRouter client
    
    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter base URL
        
    Returns:
        Configured OpenRouterClient instance
    """
    return OpenRouterClient(api_key, base_url)