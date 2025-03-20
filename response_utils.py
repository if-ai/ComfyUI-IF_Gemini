import torch
from io import BytesIO

from .image_utils import tensor_to_pil, resize_image, sample_video_frames
from .audio_utils import process_audio

def prepare_response(prompt, input_type, text_input=None, images=None, video=None, audio=None, max_images=6):
    """Prepare content based on input type for Gemini API"""
    # Auto-detect input type if needed
    detected_type = input_type
    
    # Override input_type based on provided inputs
    if input_type == "text" and images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
        detected_type = "image"
    elif video is not None and isinstance(video, torch.Tensor) and video.nelement() > 0:
        detected_type = "video"
    elif audio is not None:
        detected_type = "audio"
        
    # Process based on detected type
    if detected_type == "text":
        text_content = prompt if not text_input else f"{prompt}\n{text_input}"
        return [{"text": text_content}]
            
    elif detected_type == "image":
        # Handle multiple images input
        all_images = []
        
        # Process images if provided
        if images is not None:
            # Check if images is a tensor with batch dimension
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 4:  # [batch, H, W, C]
                    # Limit number of images to max_images
                    num_images = min(images.shape[0], max_images)
                    
                    for i in range(num_images):
                        pil_image = tensor_to_pil(images[i])
                        pil_image = resize_image(pil_image, 1568)
                        all_images.append(pil_image)
                else:  # Single image tensor [H, W, C]
                    pil_image = tensor_to_pil(images)
                    pil_image = resize_image(pil_image, 1568)
                    all_images.append(pil_image)
            # Handle case where images is a list
            elif isinstance(images, list):
                for img_tensor in images[:max_images]:  # Limit to max_images
                    pil_image = tensor_to_pil(img_tensor)
                    pil_image = resize_image(pil_image, 1568)
                    all_images.append(pil_image)
                    
        # If we have any images, create the parts structure
        if all_images:
            # Modify prompt to handle multiple images
            if len(all_images) > 1:
                modified_prompt = f"Analyze these {len(all_images)} images. {prompt} Please describe each image separately."
            else:
                modified_prompt = prompt
                
            parts = [{"text": modified_prompt}]
            
            for idx, img in enumerate(all_images):
                # Convert image to bytes
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_byte_arr
                    }
                })
            
            return [{"parts": parts}]
        else:
            raise ValueError("No valid images provided")
            
    elif detected_type == "video" and video is not None:
        # Handle video input (sequence of frames)
        frames = sample_video_frames(video)
        if frames:
            # Convert frames to proper format
            parts = [{"text": f"Analyzing video frames. {prompt}"}]
            for frame in frames:
                # Convert each frame to bytes
                img_byte_arr = BytesIO()
                frame.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_byte_arr
                    }
                })
            return [{"parts": parts}]
        else:
            raise ValueError("Invalid video format")
                
    elif detected_type == "audio" and audio is not None:
        audio_bytes = process_audio(audio)
        
        return [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_bytes
                    }
                }
            ]
        }]
    else:
        raise ValueError(f"Invalid or missing input for {detected_type}")