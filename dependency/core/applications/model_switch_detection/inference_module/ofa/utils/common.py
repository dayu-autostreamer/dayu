import torch

def resize_images(image_list):
    """
    Resize images in a list to the same size (using the max height/width as reference), then stack them into a batch.

    Args:
        image_list: List[torch.Tensor]. Each tensor has shape [C, H, W]; H and W may differ.

    Returns:
        torch.Tensor of shape [B, C, H_max, W_max], where B is the list length.
    """
    if not image_list:
        raise ValueError("Image list is empty")
    
    # Get maximum height and width
    max_height = max(img.size(1) for img in image_list)
    max_width = max(img.size(2) for img in image_list)
    
    # Resize each image
    resized_images = []
    for img in image_list:
        if img.size(1) != max_height or img.size(2) != max_width:
            # Create a new tensor
            resized = torch.zeros(
                img.size(0), max_height, max_width,
                dtype=img.dtype, device=img.device
            )
            # Copy original data
            h, w = img.size()[1:]
            resized[:, :h, :w] = img
            resized_images.append(resized)
        else:
            resized_images.append(img)

    # Stack into a batch
    return torch.stack(resized_images, dim=0)