# OpenRouter Compatibility Notice

## ⚠️ Important Compatibility Issue

**TL;DR**: The Google Gemini SDK (`genai`) used by this node may not be fully compatible with OpenRouter's API, which is designed for OpenAI SDK compatibility.

## What This Means

OpenRouter provides an OpenAI-compatible API, but this ComfyUI node uses Google's Gemini SDK. This creates a fundamental compatibility mismatch that can cause:

1. **HTML Content in Logs**: Instead of API responses, you may see HTML webpage content
2. **API Key Validation Failures**: Valid OpenRouter keys may fail validation  
3. **Connection Errors**: The client may try to access OpenRouter as a webpage instead of an API

## Why This Happens

- **OpenRouter**: Designed for OpenAI SDK (uses OpenAI's API format)
- **This Node**: Uses Google Gemini SDK (expects Google's API format)
- **Result**: The Gemini client doesn't understand OpenRouter's response format

## Current Status

The integration has been implemented with:
- ✅ Environment variable support for OpenRouter keys
- ✅ Base URL configuration for OpenRouter endpoint
- ✅ Provider selector for manual choice
- ✅ Reduced logging to minimize HTML spam
- ⚠️ Limited compatibility due to SDK mismatch

## Recommended Solutions

### For Reliable OpenRouter Usage

1. **Use an OpenAI SDK-based ComfyUI node** with OpenRouter instead
2. **Use direct Gemini API** with this node (which works perfectly)
3. **Wait for a dedicated OpenRouter ComfyUI node** that uses the OpenAI SDK

### For Testing/Development

If you want to experiment with the current implementation:
1. Set `api_provider` to "openrouter"  
2. Use OpenRouter API key
3. Select OpenRouter models like `google/gemini-2.5-flash-image-preview:free`
4. Expect potential issues and error messages

## Technical Details

The core issue is that OpenRouter expects requests formatted for OpenAI's API:
```python
# What OpenRouter expects (OpenAI format)
import openai
client = openai.OpenAI(base_url="https://openrouter.ai/api/v1")
```

But this node uses:
```python  
# What this node uses (Gemini format)
import google.genai as genai
client = genai.Client(base_url="https://openrouter.ai/api/v1")  # Incompatible
```

## Alternatives

For reliable access to Gemini models through third-party providers:
- Use this node with direct Google API (recommended)
- Find ComfyUI nodes that use OpenAI SDK for OpenRouter
- Use custom nodes specifically designed for OpenRouter

## Future

A proper OpenRouter integration would require either:
1. A separate node built with OpenAI SDK
2. Hybrid support in this node (complex)
3. Changes to how OpenRouter handles Gemini SDK requests (unlikely)

For now, this node works best with direct Google Gemini API access.