import json
import os
import logging
from aiohttp import web
from .env_utils import get_api_key
from dotenv import load_dotenv

# Create a logger
logger = logging.getLogger(__name__)

# Define default models when no API key is available
default_models = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-1.5-pro-latest"
]

# Try to import server components
try:
    from server import PromptServer
    
    @PromptServer.instance.routes.post("/gemini/check_api_key")
    async def check_api_key(request):
        """Check if a Gemini API key is valid"""
        try:
            # Get key from request
            data = await request.json()
            api_key = data.get("api_key", "").strip()
            
            # If no key in request, try environment
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
                
                # If still no key, try .env file
                if not api_key:
                    env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                    if env_api_key:
                        api_key = env_api_key
            
            if not api_key:
                return web.json_response({
                    "status": "error", 
                    "message": "No API key found. Please provide a key."
                })
            
            # Use the check_gemini_api_key function
            from .gemini_node import check_gemini_api_key
            is_valid, message = check_gemini_api_key(api_key)
            
            if is_valid:
                return web.json_response({
                    "status": "success", 
                    "message": message
                })
            else:
                return web.json_response({
                    "status": "error", 
                    "message": message
                })
            
        except Exception as e:
            logger.error(f"Error checking API key: {str(e)}")
            return web.json_response({
                "status": "error", 
                "message": f"Error checking API key: {str(e)}"
            })

    @PromptServer.instance.routes.post("/gemini/get_models")
    async def get_models_for_ui(request):
        """Get available Gemini models for the UI"""
        try:
            # Get API key from request
            try:
                data = await request.json()
                external_key = data.get("external_api_key", "").strip()
            except:
                external_key = ""
            
            # API key priority
            api_key = None
            
            # First priority: external key from request
            if external_key:
                api_key = external_key
            # Second priority: system environment
            elif os.environ.get("GEMINI_API_KEY"):
                api_key = os.environ.get("GEMINI_API_KEY")
            # Third priority: .env file
            else:
                env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if env_api_key:
                    api_key = env_api_key
            
            if not api_key:
                # Return default models if no API key found
                logger.warning("No API key found, returning default models list")
                return web.json_response(default_models)
            
            # Get models using the api key
            from .gemini_node import get_available_models
            gemini_models = get_available_models(api_key)
            
            return web.json_response(gemini_models)
            
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return web.json_response(default_models)

except (ImportError, AttributeError) as e:
    logger.error(f"Error importing PromptServer, API routes will not be registered: {str(e)}")