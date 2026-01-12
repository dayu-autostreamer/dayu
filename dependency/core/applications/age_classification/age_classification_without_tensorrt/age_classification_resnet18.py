import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from core.lib.common import LOGGER


class AgeClassifier(nn.Module):
    """Age classification model - ResNet18"""
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.age_classes = 10
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, self.age_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.resnet(x)
        out = self.softmax(out)
        return out


class AgeClassificationResNet18:
    """Age classifier using PyTorch ResNet18"""
    
    def __init__(self, weights, device=0):
        """
        Initialize age classifier
        
        Args:
            weights: Model weights file path
            device: GPU device ID, -1 for CPU
        """
        self.weights = weights
        self.device_id = device
        
        # Setup device
        if torch.cuda.is_available() and device >= 0:
            self.device = torch.device(f'cuda:{device}')
            self.use_cpu = False
        else:
            self.device = torch.device('cpu')
            self.use_cpu = True
        
        # Disable gradient computation
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        
        # Class labels - 10 age ranges
        self.classes = ['0-10', '11-20', '21-30', '31-40', '41-50', 
                       '51-60', '61-70', '71-80', '81-90', '91-100']
        
        # Input size
        self.input_size = 224
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = None
        self._load_model()
        
        LOGGER.info(f"AgeClassificationResNet18 initialized on device: {self.device}")
        
        # Warmup
        self._warmup()
    
    def _load_model(self):
        """Load the model"""
        try:
            LOGGER.info(f'Loading age classification model: {self.weights}')
            
            # Create model instance
            self.model = AgeClassifier()
            
            # Load weights
            if self.use_cpu:
                state_dict = torch.load(self.weights, map_location='cpu')
            else:
                state_dict = torch.load(self.weights, map_location=self.device)
            
            # Handle weight dictionary (compatible with different save formats)
            if isinstance(state_dict, dict):
                # If it's a full checkpoint, try to extract state_dict
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
            
            # Load weights into model
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            LOGGER.info('Age classification model loaded successfully')
            
        except Exception as e:
            LOGGER.error(f'Error loading age classification model: {e}')
            LOGGER.exception(e)
            raise
    
    def _warmup(self):
        """Warmup the model"""
        LOGGER.info("Warming up...")
        try:
            for _ in range(5):
                dummy_img = np.zeros([self.input_size, self.input_size, 3], dtype=np.uint8)
                self.infer(dummy_img)
            LOGGER.info("Warmup completed")
        except Exception as e:
            LOGGER.warning(f"Warmup failed: {e}")
    
    def _preprocess_image(self, raw_bgr_image):
        """
        Preprocess image for inference
        
        Args:
            raw_bgr_image: Input image in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(raw_bgr_image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        image_tensor = self.transform(image_rgb)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def infer(self, raw_image):
        """
        Perform age classification inference
        
        Args:
            raw_image: Input image, numpy array in BGR format
            
        Returns:
            Age range string: '0-10', '11-20', ..., '91-100'
        """
        try:
            if self.model is None:
                LOGGER.error("Model is not loaded")
                return '21-30'  # Return default value
            
            # Preprocess image
            input_tensor = self._preprocess_image(raw_image)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Get prediction result
            age_idx = torch.argmax(output, dim=1).item()
            
            return self.classes[age_idx]
            
        except Exception as e:
            LOGGER.warning(f"Age classification inference failed: {e}")
            return '21-30'  # Return default value

