from PIL import Image 
import numpy as np 
import torch


def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def torch_to_pil(tensor):
    if tensor.dim() == 4:
        _b, *_ = tensor.shape
        if _b == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0, ...]
    
    tensor = tensor.permute(1, 2, 0)
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor

def merge_images(images):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images)


def stack_images_horizontally(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def stack_images_vertically(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im