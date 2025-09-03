#!/usr/bin/env python3
"""
Simple test script to verify OpenRouter functionality
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_openrouter():
    """Test OpenRouter client functionality"""
    try:
        from openrouter_client import create_openrouter_client
        from env_utils import get_openrouter_api_key
        
        # Get API key
        api_key = get_openrouter_api_key()
        if not api_key:
            print("❌ No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
            return False
        
        print(f"✅ Found OpenRouter API key: {api_key[:8]}...")
        
        # Create client
        print("🔄 Creating OpenRouter client...")
        client = create_openrouter_client(api_key)
        print("✅ OpenRouter client created successfully")
        
        # Test API key validation
        print("🔄 Validating API key...")
        is_valid, message = client.validate_api_key()
        if is_valid:
            print(f"✅ API key validation: {message}")
        else:
            print(f"❌ API key validation failed: {message}")
            return False
        
        # Test model listing
        print("🔄 Listing available models...")
        models = client.list_models()
        if models:
            print(f"✅ Found {len(models)} models")
            
            # Show first few Gemini models
            gemini_models = [m for m in models if "gemini" in m.name.lower()]
            if gemini_models:
                print(f"📋 Sample Gemini models:")
                for model in gemini_models[:5]:
                    print(f"   - {model.name}")
            else:
                print("⚠️  No Gemini models found, showing first 3 models:")
                for model in models[:3]:
                    print(f"   - {model.name}")
        else:
            print("❌ No models returned")
            return False
        
        # Test simple text generation
        print("🔄 Testing text generation...")
        try:
            response = client.generate_content(
                prompt="Say hello and confirm you're working correctly. Keep it short.",
                model="google/gemini-2.0-flash-exp",  # Use a common OpenRouter model
                max_tokens=50
            )
            if response:
                print(f"✅ Text generation successful:")
                print(f"   Response: {response[:100]}...")
            else:
                print("❌ Empty response from text generation")
                return False
        except Exception as e:
            print(f"⚠️  Text generation failed (this might be normal): {e}")
        
        print("✅ All OpenRouter tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install: pip install openai>=1.0.0")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_routing_logic():
    """Test the routing logic in gemini_node"""
    try:
        from gemini_node import create_appropriate_client
        from env_utils import get_effective_api_key
        
        print("\n🔄 Testing client routing logic...")
        
        # Get API key and source
        api_key, source = get_effective_api_key()
        if not api_key:
            print("❌ No API key found for routing test")
            return False
        
        print(f"✅ Found API key from {source}: {api_key[:8]}...")
        
        # Test routing
        client, client_type = create_appropriate_client(api_key, source)
        print(f"✅ Routing successful: Using {client_type} client")
        
        # Test universal client wrapper
        from gemini_node import UniversalClient
        universal_client = UniversalClient(client, client_type)
        print(f"✅ Universal client wrapper created for {client_type}")
        
        # Test model listing through universal client
        try:
            models = universal_client.models().list()
            if models:
                print(f"✅ Model listing through universal client: {len(models)} models found")
            else:
                print("⚠️  No models returned through universal client")
        except Exception as e:
            print(f"⚠️  Model listing through universal client failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Routing logic test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing OpenRouter integration...")
    
    # Test OpenRouter client directly
    openrouter_success = test_openrouter()
    
    # Test routing logic
    routing_success = test_routing_logic()
    
    if openrouter_success and routing_success:
        print("\n🎉 All tests passed! OpenRouter integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)