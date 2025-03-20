# ComfyUI-IF_Gemini
Enjoy Google Gemini API for ComfyUI generate images, transcribe audio, sumarize videos. Making a separate implemetation of my old IF_AI tools for easy installation

## Features

- **Text Generation**: Create content, answer questions, and generate creative text formats
- **Image Analysis**: Describe, analyze, and extract information from images
- **Image Generation**: Generate images with Gemini's image generation capabilities
- **Multi-Modal Input**: Combine text and images in your prompts
- **Customizable Parameters**: Control temperature, output tokens, and other generation settings
- **Chat Mode**: Maintain conversation history for interactive sessions
- **Batch Processing**: Generate multiple outputs with a single prompt

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/if-ai/ComfyUI-IF_Gemini
```

2. Install the required Python packages:
```bash
pip install google-generativeai pillow
```

3. Set up your Gemini API key:
   - Create a Google AI Studio account at [ai.google.dev](https://ai.google.dev/)
   - Get your API key from the Google AI Studio dashboard
   - Set it as an environment variable: `GEMINI_API_KEY=your_api_key_here`
   - Or provide it directly in the node interface

4. Restart ComfyUI to load the new node

## Usage

The Gemini node appears in the "ImpactFrames💥🎞️/LLM" category in the ComfyUI node browser.

### Basic Text Generation:
1. Add the Gemini node to your workflow
2. Set the operation mode to "generate_text"
3. Enter your prompt
4. Connect to a text display node to view the results

### Image Analysis:
1. Connect an image source to the Gemini node's image input
2. Set the operation mode to "analysis"
3. Enter a prompt like "Describe this image in detail"
4. Run the workflow to get a detailed description of your image

### Image Generation:
1. Set the operation mode to "generate_images"
2. Enter a detailed prompt describing the image you want
3. Optionally connect reference images
4. Configure aspect ratio and other settings
5. Connect to an image preview node to see results

## Configuration Options

- **Model Version**: Choose from available Gemini models (flash, pro, etc.)
- **Temperature**: Control randomness (0.0 to 1.0)
- **Max Output Tokens**: Limit response length
- **Seed**: Set for reproducible results
- **Batch Count**: Generate multiple outputs
- **Aspect Ratio**: Choose image dimensions
- **Chat Mode**: Enable for conversational interactions
- **Structured Output**: Request formatted responses

## Support
If you find this tool useful, please consider supporting my work by:
- Starring this repo on GitHub
- Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/@impactframes?si=DrBu3tOAC2-YbEvc)
- Follow me on X: [Impact Frames X](https://x.com/impactframesX)
- Supporting me on Ko-fi: [Impact Frames Ko-fi](https://ko-fi.com/impactframes)
- Becoming a patron on Patreon: [Impact Frames Patreon](https://patreon.com/ImpactFrames)
Thank You!

<img src="https://count.getloli.com/get/@IFAItools_comfy?theme=moebooru" alt=":IFAItools_comfy" /> 
