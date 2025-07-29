import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
import numpy as np
import cv2

class ExposureIdentificationResNet50:
    def __init__(self, weights, device='cpu'):      
        # 1. 定义模型结构，输出5类
        self.model = resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5)
        
        if isinstance(device, int):
            device = f'cuda:{device}'
        self.device = device

        # 2. 加载权重
        state_dict = torch.load(weights, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # 3. 定义类别名称
        self.classes = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
        
        # 4. 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def infer(self, img: np.ndarray) -> str:
        """
        对输入图像进行分类推理，返回预测类别名称
        :param img: 输入图像，格式为OpenCV的BGR ndarray
        :return: str，预测的类别名称
        """
        # OpenCV的BGR转PIL的RGB
        img_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 预处理
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred = outputs.argmax(dim=1).item()
        
        return self.classes[pred]
