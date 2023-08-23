import cv2
from torchvision.transforms import transforms
import torch



class Transform:
   def __init__(self):
      super(Transform,self).__init__()
      # self.tf = transforms.ToTensor()
      self.tf = transforms.Compose([
        transforms.Resize((640,640)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
      
   def transform(self,img):
      inputs = self.tf(img)
      input = torch.unsqueeze(inputs,0)
      return input
   