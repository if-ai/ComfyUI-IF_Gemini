from .gemini_node import GeminiNode
from .api_routes import *  # Import API routes

NODE_CLASS_MAPPINGS = {
    "GeminiNode": GeminiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNode": "IF LLM Gemini AI"
}

# Path to web directory relative to this file
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]