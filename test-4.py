import cv2
import numpy as np
import json


classNum = 3

a = np.array([0,1,2,2,2])
b = np.array([0,0,1,1,2])


label = classNum * a + b
count = np.bincount(label, minlength = classNum**2)
confusion_matrix = count.reshape(classNum, classNum)

print(confusion_matrix)

print(np.diag(confusion_matrix).sum() / confusion_matrix.sum())

print(np.nanmean(np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)))


MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
MIoU = np.nanmean(MIoU)
print(MIoU)

freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

print(FWIoU)

exit()

imgcv = cv2.imdecode(np.fromfile(r"D:\Datasets\camvid\labels\0001TP_006690_P.png", dtype = np.uint8), cv2.IMREAD_COLOR)

# cv2.imencode('.jpg', imgcv)[1].tofile("tmp.png")

print(imgcv)



