import re
import os

# With OpenCV
# because opencv can not redad color palette direclty, so, pip install pypng and import
import cv2
import png
import numpy
label_path = r"D:\Datasets\huahen\SegmentationClass\001.png"
png_data = png.Reader(label_path)
color_palette = png_data.read()[3]['palette']

color_palette = numpy.array(color_palette)
color_palette = numpy.reshape(color_palette,(256,1,3)).astype(numpy.uint8)

output_arr = numpy.zeros((32,22*32),dtype=numpy.uint8)
for i in range(21):
    output_arr[:,i*32:(i+1)*32]=i
output_arr[:,21*32:22*32]=255

output_img=cv2.applyColorMap(output_arr,color_palette)
output_img=cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
cv2.imwrite('output_img.png',output_img)
