'''
参考:
https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0082.html#inference-modelarts-0082
部署到华为云 ModelArts.
'''
import PIL
from PIL import Image
import numpy as np
import os
import cv2
import common
import torch
import torchvision
import argparse
from modeling.deeplab import *

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--pretrained', type=int, default = 0)
parser.add_argument('--scaleSizeValid', type=int, default = 800)
args, unparsed = parser.parse_known_args()

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

class PTVisionService(object):
    # 初始化方法
    def __init__(self, ):
        # 调用父类构造方法
        # 本地调试
        super(PTVisionService, self).__init__()
        self.model_path = ""
        self.model_name = ""

        parser = argparse.ArgumentParser(description="")
        self.args  = args
        self.backbone='resnet'
        self.out_stride=16
        self.crop_size = 2000 
        self.sync_bn = False 
        self.freeze_bn = False 
        self.loss_type = 'ce'
        self.nclass = 2
        # training hyper params
        self.epochs = 500 
        self.start_epoch = 0 
        self.batch_size = 1 
        self.test_batch_size = 8 
        self.use_balanced_weights = True 
        # optimizer params
        self.useCUDA = True
        self.gpu_ids = '0'
        self.seed = 1 
        # checking point
        self.resume = ""
        self.checkname = None 
        # finetuning pre-trained models
        self.ft = False 
        # evaluation option
        self.eval_interval = 1 
        self.no_val = False 

        self.cuda = self.useCUDA and torch.cuda.is_available()


        if self.cuda:
            try:
                self.gpu_ids = [int(s) for s in self.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if self.sync_bn is None:
            if self.cuda and len(self.gpu_ids) > 1:
                self.sync_bn = True
            else:
                self.sync_bn = False

        if self.batch_size is None:
            self.batch_size = 4 * len(self.gpu_ids)

        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size

        if self.checkname is None:
            self.checkname = 'deeplab-'+str(self.backbone)

        self.composed_transforms = torchvision.transforms.Compose([
            common.ResizeWithPadding(self.args.scaleSizeValid)
        ])

        # Define network
        self.model = DeepLab(self.args, num_classes = self.nclass, backbone = self.backbone,
                             output_stride = self.out_stride, sync_bn = self.sync_bn, freeze_bn = self.freeze_bn)

        # Using cuda
        if self.cuda:
            self.model = self.model.cuda()

        # 获得当前脚本目录
        self.dir_path = r"C:\Users\lenovo\A\deepLabV3plus\model"
        
        # 调用自定义函数加载模型
        self.resume = os.path.join(self.dir_path, "model_best.pth.tar")
        if self.resume is not None:
            if not os.path.isfile(self.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.resume))
            checkpoint = torch.load(self.resume)
            if self.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

        # 设置为训练模式
        self.model.eval()
        self.imgSize=[] # [torch.Tensor: [100, 100], [200, 400]]

    def findBoxes(self, img):
        '''
        img: [w, h, c]
        根据目标分割的结果(矩阵)找出矩形框, 目前要求矩阵数值背景为 0, 目标物体为 1.
        '''    
        #imgcv = cv2.imread("model/1.jpg", 0)
        #thresh = cv2.threshold(imgcv, 127, 1, cv2.THRESH_BINARY)[1]
        thresh = img.astype(np.uint8)
        areaThreshold = 10 # 面积阈值
        allBox = []
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > areaThreshold:
                M = cv2.moments(cnt)
                centerX = int(M['m10'] / M['m00'])
                centerY = int(M['m01'] / M['m00'])
                left, top, w, h = cv2.boundingRect(cnt)
                allBox.append([centerX, centerY, w, h])
                #cv2.circle(imgcv, (cx, cy), 5, (0, 0, 255), -1)
                #imgcv = cv2.rectangle(imgcv, (left, top), (left + w, top + h), (0, 255, 0), 2)
        #cv2.imwrite("tmp.png", thresh)
        return allBox

    ################################################################################
    # 方法重写
    # 数据预处理
    def _preprocess(self, data):
        preprocessed_data = {}
        input_batch = []
        with Image.open(r"D:\Datasets\huahen\JPEGImagesEval\326.jpg") as imgFile:
            imgFile = imgFile.convert("L") # https://blog.csdn.net/baicaiBC3/article/details/123412404
            imgFile = np.array(imgFile)
            imgFile = torch.tensor(imgFile , dtype = torch.float32)
            self.imgSize.append(imgFile.size()) # 保存大小


            if torch.cuda.is_available():
                input_img = self.composed_transforms({'image':imgFile, 'label': imgFile})['image']
                input_img = input_img.cuda()
                input_batch.append(input_img)

                torchvision.utils.save_image(input_img.unsqueeze(0), "tmp.png")

            else:
                input_img = self.composed_transforms({'image':imgFile, 'label': imgFile})['image']
                input_batch.append(input_img)

        
        input_batch_var = torch.stack(input_batch, dim = 0)
        preprocessed_data[0] = input_batch_var
        return preprocessed_data

    # 获取结果
    def _postprocess(self, data):
        results = []
        # 一般只有一个键值对
        # 多个的情况还没有遇到过
        v = torch.nn.functional.softmax(data, dim = 1)
        pred = torch.argmax(v, dim = 1, keepdim = True).float()
        pred_first = pred[0]
        #pred_first = torch.permute(pred_first, (1, 2, 0)).cpu().int()
        pred_first = pred_first.permute(1, 2, 0).cpu().int()


        boxes = self.findBoxes(pred_first.numpy())
        # 还原到输入图像中的坐标
        newBoxes = []
        rawH, rawW = self.imgSize[0]
        maxEdge = rawW if rawW>rawH else rawH
        scaleRatio = self.args.scaleSizeValid / maxEdge
        for box in boxes:
            if(rawW>rawH):
                newBoxes.append([box[0] // scaleRatio,
                                 box[1] // scaleRatio - (rawW - rawH) // 2,
                                 box[2] // scaleRatio,
                                 box[3] // scaleRatio,
                ])
            else:
                newBoxes.append([box[0] // scaleRatio - (rawH - rawW) // 2,
                                 box[1] // scaleRatio,
                                 box[2] // scaleRatio,
                                 box[3] // scaleRatio,
                ])

        print(boxes)
        print(newBoxes)
        exit()
        results.append({'detection_boxes' : newBoxes})

        if len(results)==1:
            return results[0]
        else:
            return results


test = PTVisionService()


a = test._preprocess(234)

b = test._postprocess(test.model(a[0]))

print(b)

