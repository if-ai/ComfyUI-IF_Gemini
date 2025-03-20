import torch
import torchaudio
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def process_audio(audio):
    """
    Process audio data for Gemini API
    
    Args:
        audio: Audio dictionary from ComfyUI with waveform and sample_rate
        
    Returns:
        bytes: Audio data as WAV bytes
    """
    try:
        if audio is None:
            return None
            
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Process waveform dimensions
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension if present
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if missing
        
        # Average multi-channel audio to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz for best compatibility with Gemini
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Save to buffer as WAV
        buffer = BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format="WAV")
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None