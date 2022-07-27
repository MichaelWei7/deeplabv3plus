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
    #imgcv = cv2.imread(inputImg)
    imgcv = cv2.imdecode(np.fromfile(inputImg,dtype=np.uint8), cv2.IMREAD_GRAYSCALE) # 解决中文路径问题 https://blog.csdn.net/hhhuua/article/details/83654079
    h, w = imgcv.shape


    # 计算横向截图列数
    num_column_initial = w // cropSize
    num_column = 0
    ratio_c = 0
    for i in range(1000):
        # 添加一个框, 计算比例
        num_column = i + 1 + num_column_initial
        ratio_c = (num_column * cropSize - w) / (num_column - 1) / (cropSize)
        if (ratio_c > min_ratio):
            break

    # 计算纵向截图行数
    num_row_initial = h // cropSize
    num_row = 0
    ratio_r = 0
    for i in range(1000):
        # 添加一个框, 计算比例
        num_row = i + 1 + num_row_initial
        ratio_r = (num_row * cropSize - h) / (num_row - 1) / (cropSize)
        if (ratio_r > min_ratio):
            break

    # 开始截图
    step_column = cropSize * (1 - ratio_c) # 步长
    step_row = cropSize * (1 - ratio_r)
    for i in range(num_column):
        for j in range(num_row):
            top_x, top_y = i * step_column, j * step_row
            bottom_x, bottom_y = top_x + cropSize, top_y + cropSize
            # 文件名: 前缀-分割行数-分割列数-行索引-列索引
            cv2.imwrite(os.path.join(new_images_dir, "%s-%s-%s-%s-%s.jpg"%(prefix, str(num_row).zfill(2), str(num_column).zfill(2), str(j+1).zfill(2), str(i+1).zfill(2))), imgcv[int(top_y):int(bottom_y),  int(top_x):int(bottom_x)])

def main():
    # 大图目录
    images_dir     = r'D:\Datasets\huahen\JPEGImagesEval-灰度-均衡化'
    # 裁剪保存目录
    new_images_dir = r'D:\Datasets\huahen\JPEGImagesEval'

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
