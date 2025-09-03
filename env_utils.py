import os
import logging
from pathlib import Path
import sys

# Set up logging with reasonable verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Reduced from DEBUG to INFO

# Add a console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Reduced verbosity
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Try to load dotenv module safely
try:
    import dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    # dotenv is optional, log a message but continue
    logger.warning("dotenv module not found. Environment variables will still work if set manually.")
    DOTENV_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error importing dotenv module: {e}")
    DOTENV_AVAILABLE = False

# Try to load .env file if dotenv is available
if DOTENV_AVAILABLE:
    try:
        dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) / '.env'
        if dotenv_path.exists():
            dotenv.load_dotenv(dotenv_path)
            logger.debug(f"Loaded .env file from {dotenv_path}")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")

def get_api_key(env_var_name, service_name):
    """
    Get API key from environment variables or config files
    
    Args:
        env_var_name: Name of the environment variable
        service_name: Name of the service (for logging)
        
    Returns:
        API key string or empty string if not found
    """
    # First check if the key is already in environment variables
    api_key = os.environ.get(env_var_name, "")
    
    if api_key:
        logger.info(f"Found {service_name} API key in environment variables")
        return api_key
    
    # Try to load from shell config files (zshrc, bashrc)
    home_dir = os.path.expanduser("~")
    shell_config_files = [
        os.path.join(home_dir, ".zshrc"),
        os.path.join(home_dir, ".bashrc"),
        os.path.join(home_dir, ".bash_profile")
    ]
    
    for config_file in shell_config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    # Look for export VAR=value or VAR=value patterns
                    import re
                    patterns = [
                        rf'export\s+{env_var_name}=[\'\"]?([^\s\'\"]+)[\'\"]?',
                        rf'{env_var_name}=[\'\"]?([^\s\'\"]+)[\'\"]?'
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            api_key = matches[0]
                            logger.info(f"Found {service_name} API key in {os.path.basename(config_file)}")
                            # Set this key to environment so it's available to other parts of the program
                            os.environ[env_var_name] = api_key
                            return api_key
            except Exception as e:
                logger.error(f"Error reading {config_file}: {str(e)}")
    
    # If dotenv is not available, we can't load from .env files
    if not DOTENV_AVAILABLE:
        return ""
    
    # List only essential .env file locations to check
    possible_locations = []
    
    # Script directory (custom node directory)
    script_dir = Path(__file__).parent
    possible_locations.append((script_dir / ".env", "custom node directory"))
    
    # One level up (custom_nodes/ComfyUI-IF_Gemini)
    parent_dir = script_dir.parent
    possible_locations.append((parent_dir / ".env", "node package directory"))
    
    # ComfyUI root directory
    try:
        comfy_root = Path.cwd()
        while comfy_root.name and comfy_root.name != "ComfyUI" and comfy_root.parent != comfy_root:
            comfy_root = comfy_root.parent
        possible_locations.append((comfy_root / ".env", "ComfyUI root directory"))
    except Exception as e:
        logger.warning(f"Error determining ComfyUI root directory: {e}")
    
    # Note: Environment variables are available for debugging if needed
    
    # Try each location - if .env files don't exist, that's fine
    env_files_found = False
    for env_path, location_name in possible_locations:
        if env_path.exists():
            env_files_found = True
            try:
                dotenv.load_dotenv(env_path)
                # Check again after loading
                api_key = os.environ.get(env_var_name, "")
                if api_key:
                    logger.info(f"Loaded {service_name} API key from .env file in {location_name}")
                    return api_key
            except Exception as e:
                logger.error(f"Error loading .env from {location_name}: {str(e)}")
    
    # If we get here, no API key was found
    if not env_files_found:
        logger.info(f"No .env files found. Using system environment or external API key for {service_name}.")
    return ""

def get_base_url():
    """
    Get custom base URL for Gemini API from environment variables
    
    Supports both generic GEMINI_BASE_URL and OpenRouter-specific configuration.
    For OpenRouter, use OPENROUTER_API_KEY and optionally set GEMINI_BASE_URL
    to "https://openrouter.ai/api/v1" or set OPENROUTER_PROXY=true.
    
    Returns:
        Base URL string or None if not configured
    """
    # Check for explicit base URL first
    base_url = os.environ.get("GEMINI_BASE_URL", "")
    
    if base_url:
        logger.info(f"Using custom Gemini base URL: {base_url}")
        return base_url
    
    # Check if OpenRouter proxy mode is enabled
    openrouter_proxy = os.environ.get("OPENROUTER_PROXY", "").lower() in ("true", "1", "yes", "on")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    if openrouter_proxy or openrouter_key:
        openrouter_base_url = "https://openrouter.ai/api/v1"
        logger.info(f"Using OpenRouter proxy base URL: {openrouter_base_url}")
        return openrouter_base_url
    
    # Check shell config files for base URL and OpenRouter settings
    home_dir = os.path.expanduser("~")
    shell_config_files = [
        os.path.join(home_dir, ".zshrc"),
        os.path.join(home_dir, ".bashrc"),
        os.path.join(home_dir, ".bash_profile")
    ]
    
    for config_file in shell_config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    import re
                    
                    # Check for explicit base URL first
                    patterns = [
                        r'export\s+GEMINI_BASE_URL=[\'\"]?([^\s\'\"]+)[\'\"]?',
                        r'GEMINI_BASE_URL=[\'\"]?([^\s\'\"]+)[\'\"]?'
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            base_url = matches[0]
                            logger.info(f"Found custom Gemini base URL in {os.path.basename(config_file)}: {base_url}")
                            os.environ["GEMINI_BASE_URL"] = base_url
                            return base_url
                    
                    # Check for OpenRouter configuration
                    openrouter_patterns = [
                        r'export\s+OPENROUTER_PROXY=[\'\"]?(true|1|yes|on)[\'\"]?',
                        r'OPENROUTER_PROXY=[\'\"]?(true|1|yes|on)[\'\"]?',
                        r'export\s+OPENROUTER_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?',
                        r'OPENROUTER_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?'
                    ]
                    for pattern in openrouter_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            if "PROXY" in pattern:
                                openrouter_base_url = "https://openrouter.ai/api/v1"
                                logger.info(f"Found OpenRouter proxy enabled in {os.path.basename(config_file)}")
                                os.environ["OPENROUTER_PROXY"] = "true"
                                return openrouter_base_url
                            else:  # OPENROUTER_API_KEY found
                                openrouter_base_url = "https://openrouter.ai/api/v1"
                                logger.info(f"Found OpenRouter API key in {os.path.basename(config_file)}, using OpenRouter proxy")
                                os.environ["OPENROUTER_API_KEY"] = matches[0]
                                return openrouter_base_url
            except Exception as e:
                logger.error(f"Error reading {config_file}: {str(e)}")
    
    # Check .env files if dotenv is available
    if DOTENV_AVAILABLE:
        possible_locations = []
        
        # Script directory (custom node directory)
        script_dir = Path(__file__).parent
        possible_locations.append((script_dir / ".env", "custom node directory"))
        
        # One level up (custom_nodes/ComfyUI-IF_Gemini)
        parent_dir = script_dir.parent
        possible_locations.append((parent_dir / ".env", "node package directory"))
        
        # ComfyUI root directory
        try:
            comfy_root = Path.cwd()
            while comfy_root.name and comfy_root.name != "ComfyUI" and comfy_root.parent != comfy_root:
                comfy_root = comfy_root.parent
            possible_locations.append((comfy_root / ".env", "ComfyUI root directory"))
        except Exception as e:
            logger.warning(f"Error determining ComfyUI root directory: {e}")
        
        for env_path, location_name in possible_locations:
            if env_path.exists():
                try:
                    dotenv.load_dotenv(env_path)
                    
                    # Check for explicit base URL first
                    base_url = os.environ.get("GEMINI_BASE_URL", "")
                    if base_url:
                        logger.info(f"Loaded custom Gemini base URL from .env file in {location_name}: {base_url}")
                        return base_url
                    
                    # Check for OpenRouter configuration
                    openrouter_proxy = os.environ.get("OPENROUTER_PROXY", "").lower() in ("true", "1", "yes", "on")
                    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
                    
                    if openrouter_proxy or openrouter_key:
                        openrouter_base_url = "https://openrouter.ai/api/v1"
                        logger.info(f"Loaded OpenRouter configuration from .env file in {location_name}")
                        return openrouter_base_url
                        
                except Exception as e:
                    logger.error(f"Error loading .env from {location_name}: {str(e)}")
    
    # No custom base URL configured - using default Google API endpoint
    return None

def get_openrouter_api_key():
    """
    Get OpenRouter API key from environment variables
    
    Returns:
        OpenRouter API key string or empty string if not found
    """
    return get_api_key("OPENROUTER_API_KEY", "OpenRouter")

def get_effective_api_key():
    """
    Get the effective API key to use based on configuration
    
    Returns priority:
    1. OpenRouter API key if OpenRouter base URL is configured
    2. Regular Gemini API key otherwise
    
    Returns:
        tuple: (api_key, source_type) where source_type is "openrouter" or "gemini"
    """
    base_url = get_base_url()
    
    # Check for OpenRouter key first regardless of base URL
    openrouter_key = get_openrouter_api_key()
    gemini_key = get_api_key("GEMINI_API_KEY", "Gemini")
    
    # If using OpenRouter base URL, prefer OpenRouter API key
    if base_url and "openrouter.ai" in base_url:
        if openrouter_key:
            logger.info("Using OpenRouter API key with OpenRouter base URL")
            return openrouter_key, "openrouter"
        elif gemini_key:
            logger.warning("Using Gemini API key with OpenRouter base URL (this may not work)")
            return gemini_key, "gemini"
    else:
        # No custom base URL or non-OpenRouter base URL
        # Prefer Gemini key for standard endpoint
        if gemini_key:
            logger.info("Using Gemini API key with standard endpoint")
            return gemini_key, "gemini"
        elif openrouter_key:
            logger.info("Using OpenRouter API key with standard endpoint (may not work without OpenRouter URL)")
            return openrouter_key, "openrouter"
    
    # Log what keys were found for debugging
    if not openrouter_key and not gemini_key:
        logger.warning("No API keys found in environment (OPENROUTER_API_KEY or GEMINI_API_KEY)")
    elif not openrouter_key:
        logger.info("GEMINI_API_KEY found but OPENROUTER_API_KEY missing")
    elif not gemini_key:
        logger.info("OPENROUTER_API_KEY found but GEMINI_API_KEY missing")
    
    return "", "none"