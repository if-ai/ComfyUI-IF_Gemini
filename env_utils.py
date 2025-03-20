import os
import dotenv
from pathlib import Path

# Try to load .env file if it exists
try:
    dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) / '.env'
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
except ImportError:
    # dotenv is optional, print a message but continue
    print("dotenv module not found. Environment variables will still work if set manually.")
except Exception as e:
    print(f"Error loading .env file: {e}")

def get_api_key(key_name, provider=None):
    """
    Get API key from environment variables with fallback handling
    
    Args:
        key_name: The environment variable name to check
        provider: Optional provider name for logging
        
    Returns:
        API key string or None if not found
    """
    # First check for specific provider API key
    api_key = os.environ.get(key_name)
    
    # If not found, print a message
    if not api_key:
        provider_name = f" for {provider}" if provider else ""
        print(f"No API key found{provider_name}. Please set {key_name} in environment variables or provide it in the node.")
    
    return api_key