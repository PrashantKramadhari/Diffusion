from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        # TODO: Implement forward_mapping
        # raise NotImplementedError("forward_mapping is not implemented yet.")
        """
        Apply 90-degree clockwise rotation to the input tensor.
        
        Args:
            im: Input tensor of shape (C, H, W) or (B, C, H, W)
            background: Not used in this implementation
            **kwargs: Additional arguments (not used)
            
        Returns:
            Rotated tensor with 90-degree clockwise rotation
        """
        # Rotate 90 degrees clockwise
        # This is equivalent to transposing and then flipping horizontally
        # Or using torch.rot90 with k=-1 (negative for clockwise)
        return torch.rot90(im, k=-1, dims=(-2, -1))

    def inverse_view(self, noise, background=None, **kwargs):
        # TODO: Implement inverse_mapping
        # raise NotImplementedError("inverse_mapping is not implemented yet.")
        # Rotate 90 degrees counter-clockwise to undo the clockwise rotation
        # This is equivalent to using torch.rot90 with k=1 (positive for counter-clockwise)
        return torch.rot90(noise, k=1, dims=(-2, -1))
