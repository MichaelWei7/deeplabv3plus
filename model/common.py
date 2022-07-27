import os
import cv2
import torch
import torchvision
import numpy as np

def calculate_weigths_labels(filePath, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    print('Calculating classes weights')
    for sample in dataloader:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    np.save(filePath, ret)

    return ret

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

class DatasetFolder(object):
    '''
    读取数据集图片,
    要求图片目录和标签目录下面都是图片, 这些图片是单通道的
    返回: 图片像素值范围 [0,255], 标签: 原始值
    tensor 类型

    有两种读取数据方法, 一种是遍历目录, 获取文件列表.
    第二种是提供一个字典: sampleDict, 如果提供这个列表, 则默认启动这种方法. 其格式如下:
    {'0':{'image': 'image path', 'label' : 'label path'}, '1' : ...}
    键是序号, 从 0 开始, 每个值是一个字典, 分别是图片和标签的文件路径.
    '''
    def __init__(self, args, imageDir = None, lableDir = None, transform = None, sampleDict = None):
        self.args = args
        self.transform = transform

        self.images = []
        self.labels = []

        if sampleDict == None:
            self.images = getFileList(imageDir)
            self.labels = getFileList(lableDir)
        else:
            for k in sampleDict:
                v = sampleDict[k]
                self.images.append(v['image'])
                self.labels.append(v['label'])

        print("Total images: ", len(self.images))
        print("Total labels: ", len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.labels[index], cv2.IMREAD_UNCHANGED)
        image = torch.tensor(image, dtype = torch.float32)
        label = torch.tensor(label, dtype = torch.float32)
        return  self.transform({'image':image, 'label': label})

class RandomCrop(torch.nn.Module):
    '''
    随机裁剪:
    返回: {'image':像素值范围 [0,255], 'label':像素值范围: 原始}
    '''
    def __init__(self):
        super().__init__()
    def forward(self, sampleDict):
        image = sampleDict['image']
        label = sampleDict['label']
        w, h = image.size()
        minEdge = w if w < h else h
        newSize = (0.8 * torch.rand(1)[0] + 0.2) * minEdge
        
        offsetW = torch.rand(1)[0] * (w - newSize)
        offsetH = torch.rand(1)[0] * (h - newSize)
        offsetW = offsetW.int()
        offsetH = offsetH.int()
        newSize = newSize.int()

        newImage = image[offsetW : offsetW + newSize, offsetH : offsetH + newSize]
        newLabel = label[offsetW : offsetW + newSize, offsetH : offsetH + newSize]
        return {'image':newImage, 'label':newLabel}

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, sampleDict):
        if torch.rand(1) < self.p:
            return {"image":torchvision.transforms.functional.hflip(sampleDict['image']),
                    "label":torchvision.transforms.functional.hflip(sampleDict['label'])}
        else:
            return sampleDict

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sampleDict):
        if torch.rand(1) < self.p:
            return {"image":torchvision.transforms.functional.vflip(sampleDict['image']),
                    "label":torchvision.transforms.functional.vflip(sampleDict['label'])}
        else:
            return sampleDict

class Resize(torch.nn.Module):
    '''
    缩放, 注意输入也是方形的图像
    返回: {'image':像素值范围 [0,1], 'label':像素值范围: 原始}
    '''
    def __init__(self, scaleSize):
        super().__init__()
        self.imageResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.BILINEAR, antialias = True)
        self.labelResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.NEAREST)

    def forward(self, sampleDict):
        image = sampleDict['image'].unsqueeze(0)
        label = sampleDict['label'].unsqueeze(0)

        image = image.repeat(3, 1, 1) / 255 # 复制成 3 通道的

        return  {'image':self.imageResize(image) , 'label':self.labelResize(label).long()}


class ResizeWithPadding(torch.nn.Module):
    '''
    缩放, 用于验证集, 加黑边
    返回: {'image':像素值范围 [0,1], 'label':像素值范围: 原始}
    '''
    def __init__(self, scaleSize):
        super().__init__()
        self.imageResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.BILINEAR, antialias = True)
        self.labelResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.NEAREST)

    def forward(self, sampleDict):
        image = sampleDict['image'].unsqueeze(0)
        label = sampleDict['label'].unsqueeze(0)

        _, w, h = image.size()

        oneSide = 0
        anoSide = 0
        padding = []

        offset = w - h
        if offset > 0:
            oneSide = offset // 2
            anoSide = oneSide + offset % 2
            padding = [oneSide, 0, anoSide, 0]
        else:
            offset = - offset
            oneSide = offset // 2
            anoSide = oneSide + offset % 2
            padding = [0, oneSide, 0, anoSide]

        image = torchvision.transforms.functional.pad(image, padding)
        label = torchvision.transforms.functional.pad(label, padding)
        image = image.repeat(3, 1, 1) / 255 # 复制成 3 通道的

        return  {'image':self.imageResize(image) , 'label':self.labelResize(label).long()}



