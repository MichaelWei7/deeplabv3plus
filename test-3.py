import os
import sys
import cv2
import numpy as np


def get_file_list(path, is_recursion = True):
    # 获取目录下的所有文件, is_recursion 表明是否递归地进行, 不递归则只获得根目录下的文件
    file_list = []
    for (root, dirs, files) in os.walk(path):
        for j in files:
            file_list.append(root + "/" + j)
        if(is_recursion == False):
            break
    file_list.sort()
    return file_list



foler = r"C:\Users\lenovo\A\hiacent\20220620-正海电容屏"

fList = get_file_list(foler)

for i in fList:
    extension = i[-3:]
    if(extension == "bmp"):

        dir = os.path.dirname(i)
        fileExt = os.path.splitext(i)[1]
        fileName = os.path.basename(i).replace(fileExt, '')

        print(i)
        print(dir)
        print(fileExt)
        print(fileName)


        imgcv = cv2.imdecode(np.fromfile(i,dtype=np.uint8), cv2.IMREAD_COLOR)

        cv2.imencode('.png', imgcv)[1].tofile(os.path.join(dir, fileName+".png"))
        
        os.remove(i)