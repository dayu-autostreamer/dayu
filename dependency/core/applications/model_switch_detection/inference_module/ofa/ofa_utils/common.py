import torch
def resize_images(image_list):
    """
    将一个list中的图片resize到相同大小（以最大尺寸为准），然后再拼接成batch
    Args:
        image_list: List[torch.Tensor]，每个tensor的shape为[C, H, W]，其中H、W可能不同
    Returns:
        torch.Tensor of shape [B, C, H_max, W_max]，其中B为list长度
    """
    if not image_list:
        raise ValueError("Image list is empty")
    
    # 获取最大的高度和宽度
    max_height = max(img.size(1) for img in image_list)
    max_width = max(img.size(2) for img in image_list)
    
    # resize每张图片
    resized_images = []
    for img in image_list:
        if img.size(1) != max_height or img.size(2) != max_width:
            # 创建新的tensor
            resized = torch.zeros(img.size(0), max_height, max_width,
                                dtype=img.dtype, device=img.device)
            # 复制原始数据
            h, w = img.size()[1:]
            resized[:, :h, :w] = img
            resized_images.append(resized)
        else:
            resized_images.append(img)
    
    # 拼接成batch
    return torch.stack(resized_images, dim=0)