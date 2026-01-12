import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
import numpy as np
import cv2

class ExposureIdentificationResNet50:
    def __init__(self, weights, device=0):
        self.model = resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5)
        
        if isinstance(device, int):
            device = f'cuda:{device}'
        self.device = device

        state_dict = torch.load(weights, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        self.classes = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def infer(self, img: np.ndarray) -> str:
        img_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred = outputs.argmax(dim=1).item()
        
        return self.classes[pred]
