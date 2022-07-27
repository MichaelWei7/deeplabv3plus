import cv2
import numpy as np
import torch
import torchvision
import glob


imageResize = torchvision.transforms.Resize(18, interpolation = torchvision.transforms.InterpolationMode.BILINEAR, antialias = True)



sample = torch.rand(3, 12, 2)

print(imageResize(sample).size())


fileList = glob.glob("")
exit()

model = torchvision.models.resnet101(pretrained = False)
sample = torch.rand(6, 3, 224, 224)
output = model(sample)

