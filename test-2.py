import cv2
import numpy as np
import torchvision
import torch

# 比例转像素
#img = cv2.imread(inputImg, 0)
img = cv2.imdecode(np.fromfile(r"C:\Users\lenovo\A\hiacent\0-ModelZoo\YOLOV5\runs\4.jpg",dtype=np.uint8), cv2.IMREAD_COLOR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.tensor(img, dtype = torch.float32).permute( (2, 0, 1)) # .contiguous()
img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels = 3)

torchvision.utils.save_image(img/255, r"C:\Users\lenovo\A\hiacent\0-ModelZoo\YOLOV5\runs\detect\exp3\5.jpg")

# result_3 = cv2.GaussianBlur(result_2, (3, 3), 1)  
# result_1 = img
# result_2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(100,100)).apply(result_1)

#cv2.imencode('.png', img * 255)[1].tofile("tmp-0.png")


