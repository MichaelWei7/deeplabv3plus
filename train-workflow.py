'''
使用 modelarts 算法管理创建算法, 此文件为启动文件.
'''
import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import train

parser = argparse.ArgumentParser(description="")
parser.add_argument('--data_url', type=str, default = "./")
parser.add_argument('--train_url', type=str, default = "./")
args, unparsed = parser.parse_known_args()

# 主函数
def main():

    # 数据处理与读取
    path = args.data_url
    sampleDict = {} # 用于存储图片和文件路径  {'0':{'image': 'image path', 'label' : 'label path'}, '1' : ...}
    # 读取文件数据
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            oneline = json.loads(line)
            annotationFile = oneline['annotation'][0]['annotation-loc']
            srcImgFile = oneline['source']
            parseXML(annotationFile, srcImgFile, sampleDict)
    print(sampleDict)

    # 开始训练
    trainer = train.Trainer(sampleDict)
    trainer.args.runDir = ""

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % trainer.args.eval_interval == (trainer.args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()


    # 复制保存权重文件及推理代码
    shutil.copyfile("model_best.pth.tar", os.path.join(sys.path[0], "model/model_best.pth.tar"))
    modelDir = os.path.join(sys.path[0], "model")
    modelDirOBS = os.path.join(args.train_url, "model")
    
    # 复制文件到OBS映射目录.
    shutil.copytree(modelDir, modelDirOBS)


class MakeClassID(object):
    """自动为类别名排序, 返回id, id 为 0 默认为背景类"""
    def __init__(self):
        super(MakeClassID, self).__init__()
        self.allClassName = {} # 存储类别名称及其对应的 ID
        self.classNum = 0 
    def getClassID(self, className : str):
        if className not in self.allClassName:
            self.allClassName[className] = self.classNum + 1
            self.classNum += 1
            return self.classNum
        else:
            return self.allClassName[className]

# 自动分配类别 ID
makeClassID = MakeClassID()

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

# 解析 xml文件并保存标签图片
def parseXML(xmlFile, srcImgPath, sampleDict):
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    fileName = root.find('filename').text
    w = int(root.find('size/width').text)
    h = int(root.find('size/height').text)
    # 创建空白图片
    img = np.zeros((h, w), dtype=np.uint8)

    # 遍历每一个标注多边形
    for polygons in root.iter('object'):
        isExisted = polygons.find('polygon')
        if(isExisted == None):
            continue
        # 找到标签类名
        className = polygons.find('name').text
        classID = makeClassID.getClassID(className)
        points = []
        # 遍历 polygon 中的每一个坐标值
        for pp in polygons.iter("polygon"):
            for i in list(pp):
                points.append(int(i.text))

        # 汇总结果
        points = np.reshape(np.array(points), (-1, 2))
        # 填充
        cv2.fillConvexPoly(img, points, classID)

    # 保存文件
    f, e = os.path.splitext(fileName)
    labelFilePath = os.path.join(f) + ".png"
    cv2.imwrite(labelFilePath, img)

    # 保存到字典中
    sampleDict[str(len(sampleDict))] = {'image' : srcImgPath, 'label' : labelFilePath}

if __name__ == "__main__":
   main()
