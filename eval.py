import argparse
import os
import numpy as np
import common
import torchvision
import glob
import cv2
from modeling.deeplab import *
import torchvision
torch.set_printoptions(edgeitems = 10, linewidth = 200)

def main():
    trainer = Trainer()
    trainer.validation()

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return r"D:\Datasets\huahen"  # folder that contains VOCdevkit/.
        if dataset == 'pascal-eval':
            return r"D:\Datasets\huahen"  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

class Trainer(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Evaluation")
        # directory
        parser.add_argument('--evalImgDir', type=str, default = r"datasets\validImg")
        parser.add_argument('--outputDir', type=str, default = r"runs")
        parser.add_argument('--classNum', type=int, default = 2)
        parser.add_argument('--scaleSize', type=int, default=500,   help='crop image size')
        parser.add_argument('--scaleSizeValid', type=int, default = 1000,   help='crop image size')
        parser.add_argument('--batch_size', type=int, default = 1,    metavar='N', help='input batch size for training (default: auto)')
        parser.add_argument('--pretrained', type=int, default = 0)
        parser.add_argument('--resume', type=str, default="model_best.pth.tar", help='put the path to resuming file if needed')

        # other
        parser.add_argument('--backbone', type=str, default='resnet',    choices=['resnet', 'xception', 'drn', 'mobilenet'], help='backbone name (default: resnet)')
        parser.add_argument('--out_stride', type=int, default=16,   help='network output stride (default: 8)')
        parser.add_argument('--dataset', type=str, default='pascal',    choices=['pascal', 'coco', 'cityscapes'],   help='dataset name (default: pascal)')
        parser.add_argument('--use_sbd', action='store_true', default=False,    help='whether to use SBD dataset (default: True)')
        parser.add_argument('--workers', type=int, default=4,   metavar='N', help='dataloader threads')
        parser.add_argument('--base_size', type=int, default=500,   help='base image size')
        parser.add_argument('--crop_size', type=int, default=500,   help='crop image size')
        parser.add_argument('--sync_bn', type=bool, default=False,   help='whether to use sync bn (default: auto)')
        parser.add_argument('--freeze_bn', type=bool, default=False,    help='whether to freeze bn parameters (default: False)')
        parser.add_argument('--loss_type', type=str, default='ce',  choices=['ce', 'focal'], help='loss func type (default: ce)')
        # training hyper params
        parser.add_argument('--start_epoch', type=int, default=0,   metavar='N', help='start epochs (default:0)')
        parser.add_argument('--test_batch_size', type=int, default=2,  metavar='N', help='input batch size for testing (default: auto)')
        parser.add_argument('--use_balanced_weights', action='store_true', default=True,   help='whether to use balanced weights (default: False)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,  metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--nesterov', action='store_true', default=False,   help='whether use nesterov (default: False)')
        # cuda, seed and logging
        parser.add_argument('--useCUDA', action='store_true', default = True)
        parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a \   comma-separated list of integers only (default=0)')
        # checking point
        parser.add_argument('--checkname', type=str, default=None,  help='set the checkpoint name')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=True, help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval_interval', type=int, default=1, help='evaluuation interval (default: 1)')
        parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')
        self.args, _ = parser.parse_known_args()

        # cuda设置
        self.args.cuda = self.args.useCUDA and torch.cuda.is_available()
        if self.args.cuda:
            try:
                self.args.gpu_ids = [int(s) for s in self.args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if self.args.sync_bn is None:
            if self.args.cuda and len(self.args.gpu_ids) > 1:
                self.args.sync_bn = True
            else:
                self.args.sync_bn = False

        # batch 设置
        if self.args.batch_size is None:
            self.args.batch_size = 4 * len(self.args.gpu_ids)
        if self.args.test_batch_size is None:
            self.args.test_batch_size = self.args.batch_size

        # 图像预处理
        self.datasetTransforms = {
            'valid': torchvision.transforms.Compose([
                common.Resize(self.args.scaleSizeValid),
                common.Normalize(classNum = self.args.classNum)
            ]),
        }

        # 加载数据集
        sampleDict = []

        # 获取训练集字典列表
        images = glob.glob(f"{self.args.evalImgDir}/*")
        assert len(images) != 0
        for i in range(len(images)):
            sampleDict.append({'image':images[i], 'label':images[i]})

        self.imageFolder = {
            'valid' : common.DatasetFolder(self.args, transform = self.datasetTransforms['valid'], sampleDict = sampleDict)
        }
        self.datasetLoader = {
            'valid' : torch.utils.data.DataLoader(self.imageFolder['valid'], batch_size = self.args.batch_size, shuffle=False, pin_memory = False)
        }

        # Define network
        model = DeepLab(self.args,
                        num_classes=self.args.classNum,
                        backbone=self.args.backbone,
                        output_stride=self.args.out_stride,
                        sync_bn=self.args.sync_bn,
                        freeze_bn=self.args.freeze_bn)

        # Define Criterion
        # whether to use class balanced weights
        self.model = model
        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))

            checkpoint = torch.load(self.args.resume, map_location = torch.device("cpu"))

            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            if not self.args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            self.args.start_epoch = 0

        self.model.eval()

    def validation(self):
        for i, sample in enumerate(self.datasetLoader['valid']):
            print(i)
            images = sample['image']
            if self.args.cuda:
                images = images.cuda()
            with torch.no_grad():
                output = self.model(images)

            _, _, h, w = images.size()
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            torchvision.utils.save_image(torch.Tensor(pred).unsqueeze(1), f"{self.args.outputDir}{os.sep}{str(i).zfill(3)}.png")
            
            boxes = findBoxes(torch.Tensor(pred).permute(1, 2, 0).cpu().int().numpy(), w, h)
            print(boxes)

    # 服务部署
    def deploy(self, imagePath):
        sampleDict = [ {'image':imagePath, 'label':imagePath} ] # 字典列表
        
        self.imageFolder = {
            'valid' : common.DatasetFolder(self.args, transform = self.datasetTransforms['valid'], sampleDict = sampleDict)
        }
        self.datasetLoader = {
            'valid' : torch.utils.data.DataLoader(self.imageFolder['valid'], batch_size = self.args.batch_size, shuffle=False, pin_memory = False)
        }

        for i, sample in enumerate(self.datasetLoader['valid']):
            print(i)
            images = sample['image']
            if self.args.cuda:
                images = images.cuda()
            with torch.no_grad():
                output = self.model(images)

            _, _, h, w = images.size()
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            boxes = findBoxes(torch.Tensor(pred).permute(1, 2, 0).cpu().int().numpy(), w, h)
            
            
            print(boxes)


def findBoxes(img, w, h):
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
            left, top, w_, h_ = cv2.boundingRect(cnt)
            allBox.append([centerX/w , centerY/h , w_/w, h_/h])
            #cv2.circle(imgcv, (cx, cy), 5, (0, 0, 255), -1)
            #imgcv = cv2.rectangle(imgcv, (left, top), (left + w, top + h), (0, 255, 0), 2)
    #cv2.imwrite("tmp.png", thresh)
    return allBox

if __name__ == "__main__":
   main()
