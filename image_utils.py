import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import base64
import io
import requests
import logging

logger = logging.getLogger(__name__)

def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image with improved error handling
    and proper shape processing to avoid dimension issues
    
    Args:
        tensor: A PyTorch tensor representing an image
            
    Returns:
        PIL.Image: The converted PIL image
    """
    try:
        # Ensure tensor is on CPU
        tensor = tensor.cpu()
            
        # Handle different tensor shapes properly
        if tensor.dim() == 4:  # [B, H, W, C] or [B, C, H, W]
            if tensor.shape[0] == 1:  # Batch of 1
                tensor = tensor.squeeze(0)  # Remove batch dimension
            else:
                # We'll take the first image from the batch
                tensor = tensor[0]
            
        # Handle both [C, H, W] and [H, W, C] formats
        if tensor.dim() == 3:
            if tensor.shape[0] == 3 and tensor.shape[-1] != 3:  # Channels-first format [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # Convert to [H, W, C]
            
        # Special case for shape [1, H, W] (grayscale)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [H, W]
            tensor = tensor.unsqueeze(-1)  # [H, W, 1]
            tensor = tensor.repeat(1, 1, 3)  # Convert to RGB [H, W, 3]
            
        # If we have just [H, W], add a channel dimension for grayscale
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)  # Add channel dimension [H, W, 1]
            tensor = tensor.repeat(1, 1, 3)  # Convert to RGB [H, W, 3]
            
        # Scale to 0-255 range for uint8
        numpy_array = np.clip(tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
            
        # Create PIL image (no mode specification required, numpy shape determines format)
        pil_image = Image.fromarray(numpy_array)
        return pil_image
            
    except Exception as e:
        logger.error(f"Error in tensor_to_pil: {e}")
        # Return a gray placeholder as fallback
        return Image.new('RGB', (512, 512), color=(99, 99, 99))

def prepare_batch_images(images, max_images=6, max_size=1024):
    """
    Process batch images safely for the Gemini API
    
    Args:
        images: Tensor or list of tensors
        max_images: Maximum number of images to process
        max_size: Maximum dimension size for resizing
            
    Returns:
        List of PIL images ready for API
    """
    prepared_images = []
        
    try:
        if images is None or (isinstance(images, torch.Tensor) and images.nelement() == 0):
            return []
                
        # Handle tensor with batch dimension [B, H, W, C]
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_size = min(images.shape[0], max_images)
            for i in range(batch_size):
                # Convert each image in batch to PIL
                pil_image = tensor_to_pil(images[i])
                pil_image = resize_image(pil_image, max_size)
                prepared_images.append(pil_image)
            
        # Handle single image tensor [H, W, C] or [C, H, W]
        elif isinstance(images, torch.Tensor) and images.dim() == 3:
            pil_image = tensor_to_pil(images)
            pil_image = resize_image(pil_image, max_size)
            prepared_images.append(pil_image)
                
        # Handle list of tensors
        elif isinstance(images, list):
            for img_tensor in images[:max_images]:
                pil_image = tensor_to_pil(img_tensor)
                pil_image = resize_image(pil_image, max_size)
                prepared_images.append(pil_image)
                    
    except Exception as e:
        logger.error(f"Error preparing batch images: {e}")
            
    return prepared_images


def resize_image(image, max_size=1024):
    """Resize image while preserving aspect ratio"""
    width, height = image.size
    
    # Calculate ratio to fit within max_size while maintaining aspect
    ratio = min(max_size/width, max_size/height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def resize_image_to_dimensions(image, target_width, target_height, crop_to_fit=True):
    """
    Resize image to specific dimensions, optionally cropping to fit
    
    Args:
        image (PIL.Image): Input image
        target_width (int): Target width
        target_height (int): Target height
        crop_to_fit (bool): If True, crop to fit target aspect ratio
        
    Returns:
        PIL.Image: Resized image matching target dimensions
    """
    if not isinstance(image, Image.Image):
        try:
            # Try to convert to PIL if it's a tensor
            image = tensor_to_pil(image)
        except:
            logger.error("Cannot convert input to PIL image")
            return Image.new('RGB', (target_width, target_height), color=(99, 99, 99))
            
    current_width, current_height = image.size
    current_aspect = current_width / current_height
    target_aspect = target_width / target_height
    
    if crop_to_fit and abs(current_aspect - target_aspect) > 0.05:
        # Need to crop to match target aspect ratio
        if current_aspect > target_aspect:
            # Image is wider than target, crop width
            new_width = int(current_height * target_aspect)
            left = (current_width - new_width) // 2
            image = image.crop((left, 0, left + new_width, current_height))
        else:
            # Image is taller than target, crop height
            new_height = int(current_width / target_aspect)
            top = (current_height - new_height) // 2
            image = image.crop((0, top, current_width, top + new_height))
    
    # Resize to target dimensions
    return image.resize((target_width, target_height), Image.LANCZOS)

def sample_video_frames(video_tensor, max_frames=16):
    """Sample frames from a video tensor for analysis"""
    if len(video_tensor.shape) != 4 or video_tensor.shape[0] <= 1:
        return None
    
    # Get number of frames
    num_frames = video_tensor.shape[0]
    
    # If we have fewer frames than max_frames, use all frames
    if num_frames <= max_frames:
        frames_to_sample = list(range(num_frames))
    else:
        # Sample evenly spaced frames
        frames_to_sample = [int(i * num_frames / max_frames) for i in range(max_frames)]
    
    # Convert selected frames to PIL images
    sampled_frames = []
    for i in frames_to_sample:
        frame_tensor = video_tensor[i]
        pil_image = tensor_to_pil(frame_tensor)
        pil_image = resize_image(pil_image, 1024)
        sampled_frames.append(pil_image)
    
    return sampled_frames

def load_placeholder_image(placeholder_image_path):
    # Ensure the placeholder image exists
    if not os.path.exists(placeholder_image_path):
        # Create a proper RGB placeholder image
        placeholder = Image.new('RGB', (512, 512), color=(99, 99, 99))
        os.makedirs(os.path.dirname(placeholder_image_path), exist_ok=True)
        placeholder.save(placeholder_image_path)
    
    img = Image.open(placeholder_image_path)
    
    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']
    
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]
        
        if image.size[0] != w or image.size[1] != h:
            continue
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)

def create_placeholder_image(placeholder_image_path=None):
    """Create 1024x1024 placeholder"""
    img = Image.new('RGB', (1024, 1024), color=(99, 99, 99))
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]

def create_placeholder_image_with_size(width=1024, height=1024):
    """Create placeholder with specified dimensions"""
    img = Image.new('RGB', (width, height), color=(99, 99, 99))
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]

def process_images_for_comfy(images, placeholder_image_path=None, response_key='data', field_name='b64_json', field2_name=""):
    """Process images for ComfyUI with 1024px processing and padding"""
    def _process_single_image(image, target_width=None, target_height=None):
        try:
            if image is None:
                if target_width and target_height:
                    return create_placeholder_image_with_size(target_width, target_height), torch.ones((1, target_height, target_width), dtype=torch.float32)
                else:
                    return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

            # Handle JSON/API response
            if isinstance(image, dict):
                try:
                    # Only attempt to extract from response if response_key is provided
                    if response_key and response_key in image:
                        items = image[response_key]
                        if isinstance(items, list):
                            for item in items:
                                # Only attempt to get field_name if it's provided
                                if field2_name and field_name:
                                    image_data = item.get(field2_name, {}).get(field_name)
                                elif field_name:
                                    image_data = item.get(field_name)
                                else:
                                    continue
                                
                                if image_data:
                                    # Convert the first valid image found
                                    if isinstance(image_data, str):
                                        if image_data.startswith(('data:image', 'http:', 'https:')):
                                            image = image_data  # Will be handled by URL processing below
                                        else:
                                            # Handle base64 directly
                                            image_data = base64.b64decode(image_data)
                                            image = Image.open(io.BytesIO(image_data))
                                            break
                    
                    if isinstance(image, dict):
                        logger.warning(f"No valid image found in response under key '{response_key}'")
                        if target_width and target_height:
                            return create_placeholder_image_with_size(target_width, target_height), torch.ones((1, target_height, target_width), dtype=torch.float32)
                        else:
                            return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Error processing API response: {str(e)}")
                    if target_width and target_height:
                        return create_placeholder_image_with_size(target_width, target_height), torch.ones((1, target_height, target_width), dtype=torch.float32)
                    else:
                        return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

            # Handle different image types
            if isinstance(image, bytes):
                # Handle binary data
                image = Image.open(io.BytesIO(image)).convert('RGB')
            elif isinstance(image, str):
                # Handle image URLs or base64 strings
                if image.startswith('data:image'):
                    base64_data = image.split('base64,')[1]
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                elif image.startswith(('http:', 'https:')):
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(image).convert('RGB')
            elif isinstance(image, torch.Tensor):
                # Already handled by tensor_to_pil
                image = tensor_to_pil(image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')

            # Resize to specific dimensions if provided, otherwise use standard resize
            if target_width and target_height:
                image = resize_image_to_dimensions(image, target_width, target_height)
            else:
                image = resize_image(image, 1024)
            
            # Convert PIL to tensor in ComfyUI format [1, H, W, 3]
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
            
            return img_tensor, torch.ones((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            if target_width and target_height:
                return create_placeholder_image_with_size(target_width, target_height), torch.ones((1, target_height, target_width), dtype=torch.float32)
            else:
                return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

    try:
        # Handle binary image data from API responses
        if isinstance(images, (bytes, bytearray)):
            return _process_single_image(images)
            
        # Handle API responses as dictionaries
        if isinstance(images, dict) and response_key in images:
            # Process each item in API response
            all_tensors = []
            all_masks = []
            
            items = images[response_key]
            if isinstance(items, list):
                for item in items:
                    try:
                        img_data = None
                        
                        # Extract the image data based on the field name
                        if field_name in item:
                            img_data = item[field_name]
                        elif isinstance(item, dict) and field2_name in item and field_name in item[field2_name]:
                            img_data = item[field2_name][field_name]
                            
                        if img_data:
                            # Handle base64 encoded data
                            if isinstance(img_data, str) and not img_data.startswith(('http:', 'https:', 'data:')):
                                try:
                                    img_bytes = base64.b64decode(img_data)
                                    img_tensor, mask_tensor = _process_single_image(img_bytes)
                                    all_tensors.append(img_tensor)
                                    all_masks.append(mask_tensor)
                                except Exception as e:
                                    logger.error(f"Error decoding base64 image: {str(e)}")
                            else:
                                img_tensor, mask_tensor = _process_single_image(img_data)
                                all_tensors.append(img_tensor)
                                all_masks.append(mask_tensor)
                    except Exception as e:
                        logger.error(f"Error processing response item: {str(e)}")
                        continue
                
                if all_tensors:
                    # FIX: Resize all images to the same size before batching
                    # Get the size of the first tensor to use as reference
                    target_h = all_tensors[0].shape[1]
                    target_w = all_tensors[0].shape[2] 
                    
                    # Create a list to store the resized tensors
                    resized_tensors = []
                    resized_masks = []
                    
                    for i, tensor in enumerate(all_tensors):
                        h, w = tensor.shape[1], tensor.shape[2]
                        
                        # If dimensions don't match, resize the tensor
                        if h != target_h or w != target_w:
                            # Convert tensor to PIL image for resize
                            pil_img = tensor_to_pil(tensor.squeeze(0))
                            resized_img = resize_image_to_dimensions(pil_img, target_w, target_h)
                            
                            # Convert back to tensor
                            resized_array = np.array(resized_img).astype(np.float32) / 255.0
                            resized_tensor = torch.from_numpy(resized_array)[None,]
                            
                            # Also resize the mask
                            resized_mask = torch.ones((1, target_h, target_w), dtype=torch.float32)
                            
                            resized_tensors.append(resized_tensor)
                            resized_masks.append(resized_mask)
                        else:
                            # No resize needed
                            resized_tensors.append(tensor)
                            resized_masks.append(all_masks[i])
                    
                    # Now all tensors should have the same dimensions
                    return torch.cat(resized_tensors, dim=0), torch.cat(resized_masks, dim=0)
            
            # If no valid images processed, return placeholder
            return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

        # Handle list/batch of images
        if isinstance(images, (list, tuple)):
            all_tensors = []
            all_masks = []
            
            for img in images:
                try:
                    img_tensor, mask_tensor = _process_single_image(img)
                    all_tensors.append(img_tensor)
                    all_masks.append(mask_tensor)
                except Exception as e:
                    logger.error(f"Error processing batch image: {str(e)}")
                    continue
            
            if all_tensors:
                # FIX: Use the first tensor's dimensions as the target size
                target_h = all_tensors[0].shape[1]
                target_w = all_tensors[0].shape[2]
                
                # Resize all tensors to match the first one
                resized_tensors = []
                resized_masks = []
                
                for i, tensor in enumerate(all_tensors):
                    h, w = tensor.shape[1], tensor.shape[2]
                    
                    # If dimensions don't match, resize the tensor
                    if h != target_h or w != target_w:
                        # Convert tensor to PIL image for resize
                        pil_img = tensor_to_pil(tensor.squeeze(0))
                        resized_img = resize_image_to_dimensions(pil_img, target_w, target_h)
                        
                        # Convert back to tensor
                        resized_array = np.array(resized_img).astype(np.float32) / 255.0
                        resized_tensor = torch.from_numpy(resized_array)[None,]
                        
                        # Also resize the mask
                        resized_mask = torch.ones((1, target_h, target_w), dtype=torch.float32)
                        
                        resized_tensors.append(resized_tensor)
                        resized_masks.append(resized_mask)
                    else:
                        # No resize needed
                        resized_tensors.append(tensor)
                        resized_masks.append(all_masks[i])
                
                # Now all tensors should have the same dimensions
                return torch.cat(resized_tensors, dim=0), torch.cat(resized_masks, dim=0)
            
            return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

        # Handle single image
        return _process_single_image(images)

    except Exception as e:
        logger.error(f"Error in process_images_for_comfy: {str(e)}")
        return create_placeholder_image(), torch.ones((1, 1024, 1024), dtype=torch.float32)

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor with shape [1, H, W, 3]"""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension [1, H, W, 3]
    return img_tensor

def base64_to_tensor(base64_str):
    """Convert base64 string to tensor"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split('base64,')[1]
    image_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return pil_to_tensor(pil_image)