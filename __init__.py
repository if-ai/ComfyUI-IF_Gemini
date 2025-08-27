from .gemini_node import IFGeminiAdvanced
from .api_routes import *  # Import API routes

NODE_CLASS_MAPPINGS = {
    "IFGeminiNode": IFGeminiAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IFGeminiNode": "IF Gemini"
}

# Path to web directory relative to this file
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]