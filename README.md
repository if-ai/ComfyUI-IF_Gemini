# ComfyUI-IF_Gemini
Enjoy Google Gemini API for ComfyUI generate images, transcribe audio, sumarize videos. Making a separate implemetation of my old IF_AI tools for easy installation
<img width="2559" height="1232" alt="image" src="https://github.com/user-attachments/assets/010a3937-bd79-4577-a3b4-fdf0d6f2762d" />
<img width="1920" height="1075" alt="image" src="https://github.com/user-attachments/assets/a3fb04fd-ace6-4208-8df2-6887501eb879" />

## Features

- **Text Generation**: Create content, answer questions, and generate creative text formats
- **Image Analysis**: Describe, analyze, and extract information from images
- **Image Generation**: Generate images with Gemini's image generation capabilities
- **Multi-Modal Input**: Combine text and images in your prompts
- **Customizable Parameters**: Control temperature, output tokens, and other generation settings
- **Chat Mode**: Maintain conversation history for interactive sessions
- **Batch Processing**: Generate multiple outputs with a single prompt

- ** URL PROXY **
  Users can now configure a custom Gemini API endpoint in three ways:

  1. Environment variable:
  export GEMINI_BASE_URL='https://your-proxy.com/gemini/v1'
  2. In .env file:
  GEMINI_BASE_URL=https://your-proxy.com/gemini/v1
  3. In shell config (.bashrc/.zshrc):
  export GEMINI_BASE_URL='https://your-proxy.com/gemini/v1'

## NEW OPEN ROUTER

  🚀 Usage Examples

  Method 1 - Environment Variables:
  export OPENROUTER_API_KEY="sk-or-v1-your-key"
  export OPENROUTER_PROXY="true"

  Method 2 - Direct Configuration:
  export OPENROUTER_API_KEY="sk-or-v1-your-key"
  export GEMINI_BASE_URL="https://openrouter.ai/api/v1"

  Method 3 - External API Key:
  Just paste your OpenRouter key into the external_api_key field and use OpenRouter model
  names.

  🎨 Free Image Model Access

  Use model: google/gemini-2.5-flash-image-preview:free for completely free image analysis
  through OpenRouter!

  🔧 Files Modified

  - env_utils.py - Enhanced base URL and API key detection
  - gemini_node.py - Added OpenRouter client support and model names
  - OPENROUTER_README.md - Complete documentation
  - example.env - Configuration template

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/if-ai/ComfyUI-IF_Gemini
```

2. Install the required Python packages:
```bash
cd ComfyUI-IF_Gemini
pip install -r requirements.txt
```


4. Restart ComfyUI to load the new node

## Usage

The Gemini node appears in the "ImpactFrames💥🎞️/LLM" category in the ComfyUI node browser.

3. Restart ComfyUI


1. Add your Gemini API key using one of these methods:
   - **Shell configuration file** (recommended for macOS/Linux):
     ```bash
     # In ~/.zshrc, ~/.bashrc, or ~/.bash_profile:
     export GEMINI_API_KEY=your_api_key_here
     ```
     Then restart your terminal or run `source ~/.zshrc` (or relevant file)
   
   - **System environment variable**:
     ```bash
     export GEMINI_API_KEY=your_api_key
     ```
     
   - **Directly in the node**:
     Enter your API key in the "external_api_key" field
     
   - **In a `.env` file** in the custom node directory:
     ```
     GEMINI_API_KEY=your_api_key
     ```

2. Add the "IF LLM Gemini AI" node to your workflow

3. Verify your API key using the "Verify API Key" button in the node

4. Configure the node:
   - For text generation, set "operation_mode" to "analysis" or "generate_text"
   - For image generation, set "operation_mode" to "generate_images"
   - Connect reference images (optional) for style-based generation

5. Set additional parameters as needed:
   - Prompt: Your text instructions
   - Model version: Select appropriate Gemini model
   - Temperature: Controls randomness (0.0-1.0)
   - Seed: For reproducible results

## Troubleshooting

- If you encounter API key errors, use the "Verify API Key" button to check its validity
- For image safety errors, try modifying your prompt to avoid content that may trigger safety filters
- Ensure your Gemini API has appropriate quotas for your usage

## License

MIT

## Support
If you find this tool useful, please consider supporting my work by:
- Starring this repo on GitHub
- Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/@impactframes?si=DrBu3tOAC2-YbEvc)
- Follow me on X: [Impact Frames X](https://x.com/impactframesX)
- Supporting me on Ko-fi: [Impact Frames Ko-fi](https://ko-fi.com/impactframes)
- Becoming a patron on Patreon: [Impact Frames Patreon](https://patreon.com/ImpactFrames)
Thank You!

<img src="https://count.getloli.com/get/@IFGemeini_comfy?theme=moebooru" alt=":IFGemini_comfy" /> 



