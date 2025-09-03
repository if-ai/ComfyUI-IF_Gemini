# OpenRouter Integration for ComfyUI-IF_Gemini

This ComfyUI node now supports using OpenRouter as a proxy to access Gemini models. OpenRouter provides access to multiple AI models including Google's Gemini through a unified API.

## Features Added

- **✅ Full OpenRouter Compatibility**: Uses OpenAI SDK for reliable OpenRouter communication
- **🔄 Automatic Client Switching**: Smart routing between Google and OpenAI SDKs based on configuration
- **🔧 Universal API Interface**: Transparent handling of both Gemini and OpenRouter API formats
- **🔑 Automatic API Key Detection**: Supports both `GEMINI_API_KEY` and `OPENROUTER_API_KEY`
- **📡 Smart Base URL Configuration**: Automatically configures OpenRouter endpoint when enabled
- **🏷️ Model Compatibility**: Supports both standard Gemini models and OpenRouter-specific model names
- **🔄 Non-invasive Integration**: Fully optional - existing Gemini API usage remains unchanged
- **🧪 Testing Tools**: Includes test script for verifying OpenRouter setup and functionality

## Setup Methods

### Method 1: Environment Variables (Recommended)

Set your OpenRouter API key and enable the proxy:

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Option A: Explicitly set base URL
export GEMINI_BASE_URL="https://openrouter.ai/api/v1"

# Option B: Enable automatic OpenRouter proxy
export OPENROUTER_PROXY="true"

# Optional: Set site info for OpenRouter leaderboards
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"
```

### Method 2: .env File

Create a `.env` file in your ComfyUI root or custom node directory:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
GEMINI_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=ComfyUI-IF_Gemini
```

### Method 3: External API Key (Node Input)

Simply paste your OpenRouter API key directly into the `external_api_key` field in the ComfyUI node when using OpenRouter models.

## Available OpenRouter Gemini Models

The node now includes OpenRouter-specific model names:

- `google/gemini-2.5-flash` - Standard Gemini 2.5 Flash
- `google/gemini-2.5-pro` - Gemini 2.5 Pro 
- `google/gemini-2.5-flash-image-preview` - Image analysis model
- `google/gemini-2.5-flash-image-preview:free` - Free tier image model ⭐
- `google/gemini-2.0-flash-exp` - Experimental 2.0 Flash

## Usage Examples

### Basic Text Generation with OpenRouter

**Method A: Using API Provider Selector (Easiest)**
1. Set your `OPENROUTER_API_KEY` environment variable
2. In the node, set `api_provider` to "openrouter"
3. Select any OpenRouter model (e.g., `google/gemini-2.5-flash-image-preview:free`)
4. Use the node normally - it will automatically route through OpenRouter

**Method B: Using Environment Configuration**
1. Set your `OPENROUTER_API_KEY` environment variable
2. Set `GEMINI_BASE_URL=https://openrouter.ai/api/v1` 
3. Leave `api_provider` as "auto"
4. Select any Gemini model (e.g., `google/gemini-2.5-flash-image-preview:free`)
5. Use the node normally - it will automatically route through OpenRouter

### Image Analysis with Free Gemini Model

For the new free image preview model mentioned in your request:

1. Use model: `google/gemini-2.5-flash-image-preview:free`
2. Set `api_provider` to "openrouter" (easiest)
3. Provide images in the `images` input
4. Set operation mode to "analysis" 
5. The node will use OpenRouter to access the free Gemini image model

### Manual Provider Selection

The node now includes an `api_provider` selector with three options:

- **"auto"** (default): Automatically detects based on environment configuration
- **"gemini"**: Forces use of GEMINI_API_KEY and Google's direct API
- **"openrouter"**: Forces use of OPENROUTER_API_KEY and OpenRouter proxy

## How It Works

The integration enhances the existing base URL proxy feature:

1. **API Provider Selection**: Three modes for maximum flexibility
   - **Manual**: Use `api_provider` dropdown to force "gemini" or "openrouter"
   - **Auto**: Automatically detects based on base URL configuration
   - **Fallback**: Uses any available key if preferred one is missing

2. **API Key Resolution**: Smart detection and usage
   - Manual selection: Uses specified provider's API key (GEMINI_API_KEY or OPENROUTER_API_KEY)
   - Auto mode: Prefers OpenRouter key when OpenRouter base URL is configured
   - External keys: Works with external API key field in the node

3. **Base URL Configuration**: Multiple ways to enable OpenRouter:
   - Manual: Set `api_provider` to "openrouter" (auto-enables OpenRouter URL)
   - Direct: `GEMINI_BASE_URL=https://openrouter.ai/api/v1`  
   - Auto: `OPENROUTER_PROXY=true` automatically sets OpenRouter URL
   - Auto: Having `OPENROUTER_API_KEY` enables OpenRouter mode

4. **Model Name Handling**: Supports both formats:
   - Standard: `gemini-2.5-flash`
   - OpenRouter: `google/gemini-2.5-flash-image-preview:free`

5. **Headers & Metadata**: Automatically adds OpenRouter-specific headers for tracking

6. **Improved Logging**: Reduced verbosity with clear provider selection messages

## Benefits of OpenRouter Integration

- **Access to Free Models**: Use `google/gemini-2.5-flash-image-preview:free`
- **Better Rate Limits**: OpenRouter may have different rate limiting
- **Model Availability**: Access to models that might not be available directly
- **Unified Billing**: Single account for multiple AI providers
- **Fallback Options**: Automatic model fallbacks and error handling

## Testing Your Setup

### Automated Testing

Use the included test script to verify your OpenRouter integration:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-IF_Gemini
python test_openrouter.py
```

The test script will:
- ✅ Validate your OpenRouter API key
- 📋 List available models through OpenRouter
- 🔄 Test client routing logic  
- 💬 Attempt a simple text generation request
- 🧪 Verify the universal client interface

### Manual Testing

1. **Environment Check**: Verify your environment variables are set:
   ```bash
   echo $OPENROUTER_API_KEY
   echo $GEMINI_BASE_URL
   ```

2. **ComfyUI Test**: Create a simple workflow:
   - Add an IF_Gemini node
   - Set `api_provider` to "openrouter"
   - Use model `google/gemini-2.0-flash-exp`
   - Add a simple text prompt
   - Check the console for "Using openrouter client" message

3. **Image Analysis Test**: For image models:
   - Use model `google/gemini-2.5-flash-image-preview:free`
   - Set operation_mode to "analysis"
   - Provide an image input
   - Should work without HTML errors in logs

## Backwards Compatibility

This integration is fully backwards compatible:

- Existing nodes using `GEMINI_API_KEY` continue to work unchanged
- All existing model names remain available
- Default behavior (direct Google API) is preserved
- No changes to existing workflows required

## Troubleshooting

### ✅ **Compatibility Issue Resolved**

**Update**: The OpenRouter compatibility issue has been resolved! The node now automatically detects OpenRouter configuration and uses the OpenAI SDK for reliable communication with OpenRouter's API.

**New Features**:
- Automatic client switching: OpenAI SDK for OpenRouter, Google SDK for direct Gemini API
- Universal client interface that handles both API formats transparently  
- Proper model listing and validation through OpenRouter
- Full support for image analysis and text generation

**Migration**: No changes needed - existing configurations will automatically use the improved client routing.

### "Invalid API key" errors
- Verify your OpenRouter API key format: `sk-or-v1-...`
- Use the `api_provider` selector to manually choose "openrouter" 
- Check that `GEMINI_BASE_URL` is set to OpenRouter endpoint (if using auto mode)
- Run the test script: `python test_openrouter.py` to verify setup

### Installation issues
- Make sure OpenAI SDK is installed: `pip install openai>=1.0.0`
- Restart ComfyUI after installing new dependencies

### Model not found
- Try using OpenRouter format: `google/gemini-2.5-flash-image-preview:free`
- Check OpenRouter documentation for current model availability
- Some models may be region-restricted
- Verify your OpenRouter account has sufficient credits

### Chat mode limitations
- OpenRouter integration currently uses single-message generation for compatibility
- Chat history is maintained at the application level rather than API level

### Rate limiting
- OpenRouter and Google have different rate limits
- Consider upgrading your OpenRouter plan for higher limits
- Use the free tier models to minimize costs

## OpenRouter Model Pricing

The `google/gemini-2.5-flash-image-preview:free` model is completely free to use through OpenRouter, making it perfect for testing and development.

Check [OpenRouter's pricing page](https://openrouter.ai/models) for current rates on other models.