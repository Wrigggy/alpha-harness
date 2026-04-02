import torch
from loguru import logger


def get_device(preferred: str = "auto") -> torch.device:
    """Auto-detect the best available compute device.

    Args:
        preferred: One of "auto", "cuda", "mps", "cpu".
            "auto" picks the best available in order: cuda > mps > cpu.
    """
    if preferred != "auto":
        device = torch.device(preferred)
        logger.info(f"Using requested device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device
