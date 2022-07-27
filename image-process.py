# 图像预处理
import os, csv
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

def Do(inputImg,  new_images_dir, prefix, cropSize, min_ratio):
    '''
    prefix: 文件名前缀
    '''

    # 比例转像素
    #imgcv = cv2.imread(inputImg, 0)
    imgcv = cv2.imdecode(np.fromfile(inputImg,dtype=np.uint8), cv2.IMREAD_GRAYSCALE) # 解决中文路径问题 https://blog.csdn.net/hhhuua/article/details/83654079

    # result_3 = cv2.GaussianBlur(result_2, (3, 3), 1)  

    result_1 = imgcv
    result_2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(10,10)).apply(result_1)
    

    cv2.imencode('.jpg', result_2)[1].tofile(os.path.join(new_images_dir, '%s.jpg'%(prefix, )))


def main():
    # 大图目录
    images_dir     = r'D:\Datasets\huahen\JPEGImages'
    # 裁剪保存目录
    new_images_dir = r'D:\Datasets\huahen\JPEGImages-均衡化'

    os.makedirs(new_images_dir, exist_ok=True)

    fileList = get_file_list(images_dir)
    for i in fileList:
        print(i)
        dir = os.path.dirname(i)
        fileExt = os.path.splitext(i)[1]
        fileName = os.path.basename(i).replace(fileExt, '')

        Do(i, new_images_dir, fileName, cropSize = 2000, min_ratio = 0.08)

if __name__ == "__main__":
    main()
