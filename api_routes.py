import json
from aiohttp import web
from .env_utils import get_api_key
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add routes if server is available
try:
    from server import PromptServer
    
    # Check if routes are already registered to prevent duplicate registration
    routes_registered = getattr(PromptServer.instance, '_gemini_routes_registered', False)
    
    if not routes_registered:
        @PromptServer.instance.routes.post("/gemini/check_api_key")
        async def check_api_key(request):
            """Check if a Gemini API key is valid using the check_gemini_api_key function"""
            try:
                # Get key and provider from request
                data = await request.json()
                raw_key = data.get("api_key", "")
                # Coerce to string defensively (avoid .strip on non-str)
                if isinstance(raw_key, str):
                    api_key = raw_key.strip()
                else:
                    api_key = ""
                raw_provider = data.get("api_provider", "auto")
                api_provider = raw_provider if isinstance(raw_provider, str) and raw_provider in ["auto","gemini","openrouter"] else "auto"
                
                # Priority check
                if not api_key:
                    # Check environment
                    api_key = os.environ.get("GEMINI_API_KEY", "")
                    
                    if not api_key:
                        # Check shell config files
                        home_dir = os.path.expanduser("~")
                        shell_config_files = [
                            os.path.join(home_dir, ".zshrc"),
                            os.path.join(home_dir, ".bashrc"),
                            os.path.join(home_dir, ".bash_profile")
                        ]
                        
                        import re
                        shell_key = None
                        source = None
                        
                        for config_file in shell_config_files:
                            if os.path.exists(config_file):
                                try:
                                    with open(config_file, 'r') as f:
                                        content = f.read()
                                        patterns = [
                                            r'export\s+GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?',
                                            r'GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?'
                                        ]
                                        for pattern in patterns:
                                            matches = re.findall(pattern, content)
                                            if matches:
                                                shell_key = matches[0]
                                                source = os.path.basename(config_file)
                                                # Also set in environment for future use
                                                os.environ["GEMINI_API_KEY"] = shell_key
                                                break
                                except Exception as e:
                                    logger.error(f"Error reading {config_file}: {str(e)}")
                            if shell_key:
                                api_key = shell_key
                                break
                                
                        # If still no key, check .env file
                        if not api_key:
                            env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                            if env_api_key:
                                api_key = env_api_key
                
                if not api_key:
                    return web.json_response({
                        "status": "error", 
                        "message": "No API key found. Please provide a key in the node, set it in your environment, or add it to your .env file."
                    })
                
                # Use the check_gemini_api_key function from gemini_node.py
                from .gemini_node import check_gemini_api_key
                
                # Pass the api_provider to the validation function
                is_valid, message = check_gemini_api_key(api_key, api_provider)
                
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

        @PromptServer.instance.routes.get("/gemini/available_models")
        async def get_available_models(request):
            """Get available Gemini models"""
            try:
                # Get API key with proper priority
                api_key = None
                source = None
                
                # First check system environment variables
                if os.environ.get("GEMINI_API_KEY"):
                    api_key = os.environ.get("GEMINI_API_KEY")
                    source = "system environment"
                    logger.info("Using API key from system environment variables")
                else:
                    # Check shell config files directly
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
                            logger.info(f"Checking {config_file} for API key...")
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
                        api_key = shell_key
                        source = f"{shell_source}"
                        logger.info(f"Using API key from {shell_source}")
                    else:
                        # Lastly, check .env files
                        env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                        if env_api_key:
                            api_key = env_api_key
                            source = ".env file"
                            logger.info("Using API key from .env file")
                
                if not api_key:
                    logger.warning("No Gemini API key found in any location")
                    return web.json_response({
                        "status": "error",
                        "message": "No Gemini API key found in any location (environment variables, shell configs, or .env files)",
                        "models": ["gemini-2.0-flash-exp"]  # Default fallback
                    })
                
                try:
                    # Log API key information (first few characters only for security)
                    masked_key = api_key[:5] + "..." if len(api_key) > 5 else "****"
                    logger.info(f"Using Gemini API key ({masked_key}) from {source} to fetch models")
                    
                    # Get models from the API using Client approach
                    from google import genai
                    from .gemini_node import create_appropriate_client, UniversalClient
                    
                    # Use client with configurable base URL
                    raw_client, client_type = create_appropriate_client(api_key)
                    client = UniversalClient(raw_client, client_type)
                    
                    # List available models - use the correct method for newer SDK versions
                    models = client.models().list()
                    gemini_models = []
                    
                    # Filter for Gemini models only
                    for model in models:
                        if "gemini" in model.name.lower():
                            # Extract just the model name from the full path
                            model_name = model.name.split('/')[-1]
                            gemini_models.append(model_name)
                    
                    # Always include these models even if API doesn't return them
                    default_models = [
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash-002",
                        "gemini-2.5-flash-image-preview",
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-pro",
                        "gemini-2.0-flash"
                    ]
                    
                    # Add default models if not in the list
                    for model in default_models:
                        if model not in gemini_models:
                            gemini_models.append(model)
                    
                    return web.json_response({
                        "status": "success",
                        "models": gemini_models,
                        "source": f"API key from {source}"
                    })
                    
                except Exception as e:
                    # Return default models on error
                    error_msg = str(e)
                    logger.error(f"Error fetching models: {error_msg}")
                    
                    # Provide more specific error messages for common issues
                    if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        error_detail = f"Invalid API key from {source}. Please check your API key."
                    elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                        error_detail = f"API quota exceeded for key from {source}. You've reached your usage limit."
                    else:
                        error_detail = f"Error connecting to Gemini API with key from {source}: {error_msg}"
                        
                    return web.json_response({
                        "status": "error",
                        "message": error_detail,
                        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-002",
                                   "gemini-2.5-flash-image-preview"]
                    })
                    
            except Exception as e:
                logger.error(f"Error in server: {str(e)}", exc_info=True)
                return web.json_response({
                    "status": "error", 
                    "message": f"Error in server: {str(e)}",
                    "models": ["gemini-2.5-flash", "gemini-2.5-pro"]  # Default fallback
                })

        @PromptServer.instance.routes.post("/gemini/get_models")
        async def get_models_for_ui(request):
            """Get available Gemini models specifically for the UI using the get_available_models function"""
            try:
                # Get data from request
                try:
                    data = await request.json()
                    raw_external = data.get("external_api_key", "")
                    external_key = raw_external.strip() if isinstance(raw_external, str) else ""
                    raw_provider = data.get("api_provider", None)
                    api_provider = raw_provider if isinstance(raw_provider, str) else None
                except:
                    external_key = ""
                    api_provider = None
                
                # API key priority
                api_key = None
                source = None
                
                # First priority: external key from request
                if external_key:
                    api_key = external_key
                    source = "provided in the node"
                # Second priority: system environment
                elif os.environ.get("GEMINI_API_KEY"):
                    api_key = os.environ.get("GEMINI_API_KEY")
                    source = "from environment variables"
                # Third priority: check shell config files
                else:
                    # Check shell config files directly
                    home_dir = os.path.expanduser("~")
                    shell_config_files = [
                        os.path.join(home_dir, ".zshrc"),
                        os.path.join(home_dir, ".bashrc"),
                        os.path.join(home_dir, ".bash_profile")
                    ]
                    
                    import re
                    shell_key = None
                    
                    for config_file in shell_config_files:
                        if os.path.exists(config_file):
                            try:
                                with open(config_file, 'r') as f:
                                    content = f.read()
                                    patterns = [
                                        r'export\s+GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?',
                                        r'GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?'
                                    ]
                                    for pattern in patterns:
                                        matches = re.findall(pattern, content)
                                        if matches:
                                            shell_key = matches[0]
                                            source = f"from {os.path.basename(config_file)}"
                                            # Also set in environment for future use
                                            os.environ["GEMINI_API_KEY"] = shell_key
                                            break
                            except Exception as e:
                                logger.error(f"Error reading {config_file}: {str(e)}")
                        if shell_key:
                            api_key = shell_key
                            break
                    
                    # Fourth priority: .env file
                    if not api_key:
                        env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                        if env_api_key:
                            api_key = env_api_key
                            source = "from .env file"
                
                if not api_key:
                    # Return default models if no API key found
                    return web.json_response([
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash-002",
                        "gemini-2.5-flash-image-preview"
                    ])
                
                # Use the get_available_models function from gemini_node.py
                from .gemini_node import get_available_models
                
                # Get models using the api key
                gemini_models = get_available_models(api_key, api_provider)
                
                return web.json_response(gemini_models)
                
            except Exception as e:
                logger.error(f"Error in get_models_for_ui: {str(e)}")
                return web.json_response([
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.5-flash-002",
                    "gemini-2.5-flash-image-preview"
                ])
        
        # Mark routes as registered to prevent duplicate registration
        PromptServer.instance._gemini_routes_registered = True
        logger.info("Gemini API routes registered successfully")
    else:
        logger.info("Gemini API routes already registered, skipping duplicate registration")

except (ImportError, AttributeError):
    print("PromptServer not available, skipping API routes registration")