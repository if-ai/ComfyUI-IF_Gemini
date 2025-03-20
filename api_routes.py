import json
from aiohttp import web
from .env_utils import get_api_key

# Add routes if server is available
try:
    from server import PromptServer
    
    @PromptServer.instance.routes.post("/gemini/check_api_key")
    async def check_gemini_api_key(request):
        """Check if Gemini API key is configured correctly"""
        try:
            # Try to import Google Generative AI
            api_key = get_api_key("GEMINI_API_KEY", "Gemini")
            
            if not api_key:
                return web.json_response({
                    "status": "error",
                    "message": "No Gemini API key found in environment"
                })
            
            try:
                # Test the API key by importing genai
                from google import genai
                genai.configure(api_key=api_key)
                
                # Just try to access a property to verify the config
                model_list = genai.list_models()
                
                # If we get here, the key is valid
                return web.json_response({
                    "status": "success",
                    "message": "Gemini API key is valid"
                })
                
            except Exception as e:
                return web.json_response({
                    "status": "error",
                    "message": f"Error configuring Gemini: {str(e)}"
                })
                
        except Exception as e:
            return web.json_response({
                "status": "error", 
                "message": f"Error checking API key: {str(e)}"
            })

    @PromptServer.instance.routes.get("/gemini/available_models")
    async def get_available_models(request):
        """Get available Gemini models"""
        try:
            api_key = get_api_key("GEMINI_API_KEY", "Gemini")
            
            if not api_key:
                return web.json_response({
                    "status": "error",
                    "message": "No Gemini API key found in environment",
                    "models": ["gemini-2.0-flash-exp"]  # Default fallback
                })
            
            try:
                # Get models from the API
                from google import genai
                genai.configure(api_key=api_key)
                
                models = genai.list_models()
                gemini_models = []
                
                # Filter for Gemini models only
                for model in models:
                    if "gemini" in model.name.lower():
                        # Extract just the model name from the full path
                        model_name = model.name.split('/')[-1]
                        gemini_models.append(model_name)
                
                # Add default model if not in the list
                if "gemini-2.0-flash-exp" not in gemini_models:
                    gemini_models.append("gemini-2.0-flash-exp")
                
                # Add image generation model if not in the list
                if "gemini-2.0-flash-exp-image-generation" not in gemini_models:
                    gemini_models.append("gemini-2.0-flash-exp-image-generation")
                
                return web.json_response({
                    "status": "success",
                    "models": gemini_models
                })
                
            except Exception as e:
                # Return default models on error
                return web.json_response({
                    "status": "error",
                    "message": f"Error fetching models: {str(e)}",
                    "models": ["gemini-2.0-flash-exp", "gemini-2.0-flash-exp-image-generation"]
                })
                
        except Exception as e:
            return web.json_response({
                "status": "error", 
                "message": f"Error in server: {str(e)}",
                "models": ["gemini-2.0-flash-exp"]  # Default fallback
            })

except (ImportError, AttributeError):
    print("PromptServer not available, skipping API routes registration")