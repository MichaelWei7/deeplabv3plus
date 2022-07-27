from importlib.util import LazyLoader
import os
import glob
import cv2
import torch
import torchvision
import numpy as np

def calculate_weigths_labels(filePath, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    print('Calculating classes weights...')
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
        class_weights.append(1 / (np.log(1.02 + (frequency / total_frequency))))
    ret = np.array(class_weights)
    return ret

class DatasetFolder(object):
    '''
    读取数据集图片, 转换
    
    要求: 
    1. 图片目录和标签目录下面都是图片, 这些图片是单通道的.
    2. 图片与标签图像的排列顺序要一致, 文件名可以没有对应.
    3. 图片是正常图片, 标签图片像素值为分类索引号.
    
    返回:
    图像: [batch, c, w, h], 像素值范围 [0,255], tensor 类型, float 型数值, c 通道, 单通道灰度图像会被复制成 3 通道.
    标签: [batch, 1, w, h], 标签: 原始值, tensor 类型, long 型数值

    两种读取数据方法:
    1. 提供一个字典: sampleDict, 如果此变量不为空, 则默认启动这种方法. 其格式如下:
        {'0':{'image': 'image path', 'label' : 'label path'}, '1' : ...}.
        键是序号, 从 0 开始, 每个值是一个字典, 分别是图片和标签的文件路径.
    2. 遍历目录, 获取文件列表, 然后生成 sampleDict.
    
    '''
    def __init__(self, args, imageDir = None, lableDir = None, transform = None, sampleDict = None):
        self.args = args
        self.transform = transform
        self.sampleDict = sampleDict

        # 生成图片和标签对字典
        if self.sampleDict == None:
            images = glob.glob(f"{imageDir}/*", recursive = True)
            labels = glob.glob(f"{lableDir}/*", recursive = True)
            assert len(images) == len(labels) # 确保图片和标签文件数目一样
            for i in range(len(images)):
                self.sampleDict.append({'image':images[i], 'label':labels[i]})
        print("Image number: ", len(self.sampleDict))
        
    def __len__(self):
        return len(self.sampleDict)

    def __getitem__(self, index):
        image = cv2.imread(self.sampleDict[index]['image'], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype = torch.float32).permute( (2, 0, 1)) # .contiguous()
        label = cv2.imread(self.sampleDict[index]['label'], cv2.IMREAD_GRAYSCALE)
        label = torch.tensor(label, dtype = torch.long).unsqueeze(0)

        return  self.transform({'image':image, 'label': label})

class RandomCrop(torch.nn.Module):
    '''
    随机裁剪.
    从图像中随机裁出一个矩形框子图, 框的大小按照一定规则.
    返回: {'image':像素值范围 [0,255], 'label':像素值范围: 原始 long 型}
    '''
    def __init__(self):
        super().__init__()
    def forward(self, sampleDict):
        image = sampleDict['image']
        label = sampleDict['label']
        _, w, h = image.size()
        minEdge = w if w < h else h

        # 随机采样矩形框大小
        rand_num = (torch.rand(1)[0] + 0.2) / 1.2
        newSize =  (1 - (rand_num - 1)**2 ) * minEdge
        
        # 随机采样矩形框位置
        offSetX = torch.rand(1)[0] * (w - newSize)
        offSetX = offSetX.int()

        offSetY = torch.rand(1)[0] * (h - newSize)
        offSetY = offSetY.int()

        newSize = newSize.int()

        newImage = image[:, offSetX : offSetX + newSize, offSetY : offSetY + newSize]
        newLabel = label[:, offSetX : offSetX + newSize, offSetY : offSetY + newSize]

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
    缩放, 如果输入不是正方形图片, 则缩放后的尺寸为短边尺寸
    返回: {'image':像素值范围 [0,1], 'label':像素值范围: 原始}
    '''
    def __init__(self, scaleSize):
        super().__init__()
        self.imageResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.BILINEAR, antialias = True)
        self.labelResize = torchvision.transforms.Resize(scaleSize, interpolation = torchvision.transforms.InterpolationMode.NEAREST)

    def forward(self, sampleDict):
        return  {'image':self.imageResize(sampleDict['image']) , 'label':self.labelResize(sampleDict['label'])}

class Normalize(torch.nn.Module):
    '''
    缩放, 注意输入也是方形的图像
    返回: {'image':像素值范围 [0,1], 'label':像素值范围: 原始}
    '''
    def __init__(self, classNum, mean = torch.Tensor([0, 0, 0]), std = torch.Tensor([1, 1, 1])):
        super().__init__()
        self.standardization = torchvision.transforms.Normalize(mean, std) # 标准化
        self.classNum = classNum
    def forward(self, sampleDict):
        image = sampleDict['image'] / 255
        image = self.standardization(image)

        label = sampleDict['label']
        #mask = (label >= 0) & (label < self.classNum)
        #label = label[mask].view(label.size()) 

        return  {'image': image, 'label':label}

class Padding(torch.nn.Module):
    '''
    将非正方形的图片加黑边补成正方形
    '''
    def __init__(self):
        super().__init__()

    def forward(self, sampleDict):
        image = sampleDict['image']
        label = sampleDict['label']
        _, h, w = image.size()

        oneSide = 0
        anoSide = 0
        padding = []

        offset = w - h
        if offset > 0:
            oneSide = offset // 2
            anoSide = oneSide + offset % 2
            padding = [0, oneSide, 0, anoSide]
        else:
            offset = - offset
            oneSide = offset // 2
            anoSide = oneSide + offset % 2
            padding = [oneSide, 0, anoSide, 0]

        image = torchvision.transforms.functional.pad(image, padding)
        label = torchvision.transforms.functional.pad(label, padding)

        return  {'image': image , 'label' : label}

class Evaluator(object):
    '''
        metric, 度量
    '''
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # 像素分类预测精度
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # 各个类别的预测精度的平均值, 有助于削弱数样本目多的类别对准确率的贡献, 
        # 例如背景像素数很多, 会造成虚假的高精度
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        # 平均 IOU 计算
        MIoU = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength = self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
