from .gemini_node import GeminiNode

NODE_CLASS_MAPPINGS = {
    "GeminiNode": GeminiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNode": "IF LLM Gemini AI"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]