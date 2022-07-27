from inspect import getfile
import numpy as np
import cv2, os


def getFileList(path, is_recursion = True):
    # 获取目录下的所有文件, is_recursion 表明是否递归地进行, 不递归则只获得根目录下的文件
    file_list = []
    for (root, dirs, files) in os.walk(path):
        for j in files:
            file_list.append(root + "/" + j)
        if(is_recursion == False):
            break
    file_list.sort()
    return file_list





filelist = getFileList(r"D:\Datasets\huahen\JPEGImagesEval")


for i in filelist:
    print(i)
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)

    print(img.shape)
    print(img.max())
    print(img.min())
    #cv2.imwrite(i, img)

    exit()
