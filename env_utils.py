import os
import logging
from pathlib import Path
import dotenv
import sys

# Set up logging with more detailed output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose output

# Add a console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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

def get_api_key(env_var_name, service_name):
    """
    Get API key from environment variables or .env file
    
    Args:
        env_var_name: Name of the environment variable
        service_name: Name of the service (for logging)
        
    Returns:
        API key string or empty string if not found
    """
    logger.debug(f"Looking for {service_name} API key ({env_var_name})")
    
    # First check if the key is already in environment variables
    api_key = os.environ.get(env_var_name, "")
    
    if api_key:
        logger.info(f"Found {service_name} API key in environment variables")
        return api_key
    
    # List all possible .env file locations to check
    possible_locations = []
    
    # Current working directory
    current_dir = Path.cwd()
    possible_locations.append((current_dir / ".env", "current working directory"))
    
    # Script directory and parent
    script_dir = Path(__file__).parent
    possible_locations.append((script_dir / ".env", "script directory"))
    possible_locations.append((script_dir.parent / ".env", "custom node directory"))
    
    # ComfyUI root directory (assuming typical structure)
    comfy_root = current_dir
    while comfy_root.name and comfy_root.name != "ComfyUI" and comfy_root.parent != comfy_root:
        comfy_root = comfy_root.parent
    possible_locations.append((comfy_root / ".env", "ComfyUI root directory"))
    
    # User home directory
    home_dir = Path.home()
    possible_locations.append((home_dir / ".env", "user home directory"))
    possible_locations.append((home_dir / ".comfyui.env", "user home ComfyUI config"))
    
    # Debug: Print all environment variables (keys only for security)
    logger.debug("Current environment variables: " + ", ".join(os.environ.keys()))
    
    # Try each location
    for env_path, location_name in possible_locations:
        if env_path.exists():
            logger.debug(f"Found .env file at {env_path} ({location_name})")
            try:
                dotenv.load_dotenv(env_path)
                # Check again after loading
                api_key = os.environ.get(env_var_name, "")
                if api_key:
                    logger.info(f"Loaded {service_name} API key from .env file in {location_name}")
                    return api_key
                else:
                    # Try to read the file directly to see if the key exists but isn't being loaded
                    with open(env_path, 'r') as f:
                        env_content = f.read()
                        if env_var_name in env_content:
                            logger.warning(f"{env_var_name} found in {env_path} but not loaded properly")
            except Exception as e:
                logger.error(f"Error loading .env from {location_name}: {str(e)}")
    
    # Try alternative key names (sometimes people use different formats)
    alternative_keys = [
        f"{service_name.upper()}_API_KEY",
        f"{service_name.lower()}_api_key",
        f"{service_name.upper()}_KEY",
        f"{service_name.lower()}_key"
    ]
    
    for alt_key in alternative_keys:
        if alt_key != env_var_name:  # Skip if it's the same as what we already checked
            api_key = os.environ.get(alt_key, "")
            if api_key:
                logger.info(f"Found {service_name} API key using alternative name: {alt_key}")
                return api_key
    
    # If we get here, no API key was found
    logger.warning(f"No {service_name} API key found in any location. Checked environment variables and .env files.")
    return ""