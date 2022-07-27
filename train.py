import argparse
import os
import glob
import random
import torchvision
import json
import numpy as np
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from torch.utils.tensorboard import SummaryWriter
import common
torch.set_printoptions(edgeitems = 10, linewidth = 200)
np.set_printoptions(edgeitems=10, linewidth=200)
def main():
    trainer = Trainer()

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    # 开始训练
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % trainer.args.eval_interval == (trainer.args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

class Trainer(object):
    def __init__(self, sampleDict = None):
        '''
        sampleDict: 不为空时, 自动屏蔽下面的目录. 用于适配 ModelArts 训练流程.
        '''
        parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
        # directory
        parser.add_argument('--datasetRootDir', type=str, default = r"D:\Datasets\huahen", help = "数据集根目录")
        parser.add_argument('--train_image_dir', type=str, default = r"D:\Datasets\huahen\trainImg", help = "训练集图片目录")
        parser.add_argument('--train_label_dir', type=str, default = r"D:\Datasets\huahen\trainLabel", help = "训练集标签目录")
        parser.add_argument('--valid_image_dir', type=str, default = r"", help = "验证集目录, 为空时, 自动按比例切分训练集作为验证集")
        parser.add_argument('--valid_lable_dir', type=str, default = r"", help = "验证集目录, 为空时, 自动按比例切分训练集作为验证集")
        parser.add_argument('--splitRandom', type=int, default = 0, help = "是否随机切分")
        parser.add_argument('--splitRatio', type=float, default = 0.2, help = "当 valid_image_dir 和 valid_lable_dir 为空时, 自动切分训练集作为验证集.")

        parser.add_argument('--classNum', type=int, default = 2, help="程序自动读取数据集根目录下的  datasetInfo.json 文件, 会覆盖里面的类别数目.")

        parser.add_argument('--scaleSize', type=int, default = 500,   help='crop image size')
        parser.add_argument('--scaleSizeValid', type=int, default = 500,   help='crop image size')

        parser.add_argument('--epochs', type=int, default = 3, metavar='N',   help='number of epochs to train (default: auto)')
        parser.add_argument('--batch_size', type=int, default = 6,    metavar='N', help='input batch size for training (default: auto)')
        parser.add_argument('--runDir', type=str, default = 'runs',    metavar='N', help='训练记录保存目录')
        parser.add_argument('--modelBestDir', type=str, default = './',    metavar='N', help='最好的模型保存位置')

        parser.add_argument('--pretrained', type=int, default = 0, help='use pretrained weights.')

        # other
        parser.add_argument('--backbone', type=str, default='resnet',    choices=['resnet', 'xception', 'drn', 'mobilenet'], help='backbone name (default: resnet)')
        parser.add_argument('--out_stride', type=int, default=16,   help='network output stride (default: 8)')
        parser.add_argument('--use_sbd', action='store_true', default=False,    help='whether to use SBD  (default: True)')
        parser.add_argument('--workers', type=int, default=8,   metavar='N', help='dataloader threads')
        parser.add_argument('--sync_bn', type=bool, default = False,   help='whether to use sync bn (default: auto)')
        parser.add_argument('--freeze_bn', type=bool, default = False,    help='whether to freeze bn parameters (default: False)')
        parser.add_argument('--loss_type', type=str, default='ce',  choices=['ce', 'focal'], help='loss func type (default: ce)')
        # training hyper params
        parser.add_argument('--start_epoch', type=int, default=0,   metavar='N', help='start epochs (default:0)')
        parser.add_argument('--use_balanced_weights', action='store_true', default=True,   help='whether to use balanced weights (default: False)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,  metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--nesterov', action='store_true', default=False,   help='whether use nesterov (default: False)')
        # cuda, seed and logging
        parser.add_argument('--useCUDA', type = int, default = 0, help='disables CUDA training')
        parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a \   comma-separated list of integers only (default=0)')
        # checking point
        parser.add_argument('--resume', type=str, default = None, help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default = None,  help='set the checkpoint name')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=True, help='finetuning on a different ')
        # evaluation option
        parser.add_argument('--eval_interval', type=int, default=1, help='evaluuation interval (default: 1)')
        parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')
        self.args, unparsed = parser.parse_known_args()

        # cuda设置
        if self.args.useCUDA and torch.cuda.is_available():
            try:
                self.args.gpu_ids = [int(s) for s in self.args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if self.args.sync_bn is None:
            if self.args.useCUDA and len(self.args.gpu_ids) > 1:
                self.args.sync_bn = True
            else:
                self.args.sync_bn = False

        # checkpoint 设置
        if self.args.checkname is None:
            self.args.checkname = 'deeplab-'+str(self.args.backbone)

        # Define Saver
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.writer = SummaryWriter(self.saver.experiment_dir)

        # 加载数据集
        self.datasetTransforms = {
            'train': torchvision.transforms.Compose([
                #common.RandomCrop(),
                #common.RandomVerticalFlip(),
                #common.Padding(),
                common.RandomHorizontalFlip(),
                common.Padding(),
                common.Resize(self.args.scaleSize),
                common.Normalize(classNum = self.args.classNum)
            ]),
            'valid': torchvision.transforms.Compose([
                #common.Padding(),
                common.Resize(self.args.scaleSize),
                common.Normalize(classNum = self.args.classNum)
            ]),
        }

        # 切分训练集模式
        assert (self.args.train_image_dir != "") or (self.args.train_label_dir != "" ) # 训练集不能为空
        sampleDictTrain = []
        sampleDictValid = []
        if(self.args.valid_image_dir == ""):
            # 获取训练集字典列表
            sampleDict = []
            images = glob.glob(f"{self.args.train_image_dir}/*", recursive = True)
            labels = glob.glob(f"{self.args.train_label_dir}/*", recursive = True)
            assert len(images) == len(labels) # 确保图片和标签文件数目一样
            for i in range(len(images)):
                sampleDict.append({'image':images[i], 'label':labels[i]})
            l = len(sampleDict)
            # 分割
            if(self.args.splitRandom):
                random.shuffle(sampleDict)
            sampleDictTrain = sampleDict[int(self.args.splitRatio * l) : -1]
            sampleDictValid = sampleDict[0 : int(self.args.splitRatio * l)]
        self.imageFolder = {
            'train' : common.DatasetFolder(self.args, self.args.train_image_dir, self.args.train_label_dir, self.datasetTransforms['train'], sampleDictTrain),
            'valid' : common.DatasetFolder(self.args, self.args.valid_image_dir, self.args.valid_lable_dir, self.datasetTransforms['valid'], sampleDictValid)
        }
        self.datasetLoader = {
            'train' : torch.utils.data.DataLoader(self.imageFolder['train'], batch_size = self.args.batch_size, shuffle=True,  pin_memory = True, drop_last = True),
            'valid' : torch.utils.data.DataLoader(self.imageFolder['valid'], batch_size = self.args.batch_size, shuffle=False, pin_memory = True)
        }

        # Define network
        self.model = DeepLab(self.args, 
                            num_classes=self.args.classNum,
                            backbone=self.args.backbone, output_stride=self.args.out_stride,
                            sync_bn=self.args.sync_bn, freeze_bn=self.args.freeze_bn,)
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.args.lr}, {'params': self.model.get_10x_lr_params(), 'lr': self.args.lr * 10}]
        if self.args.useCUDA:
            self.model = self.model.cuda()
        print(self.model)

        # Define Optimizer
        self.optimizer = torch.optim.SGD(train_params, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        
        # Define Criterion
        # whether to use class balanced weights
        datasetInfo = {}
        if self.args.use_balanced_weights:
            classes_weights_path = os.path.join(self.args.datasetRootDir, 'datasetInfo.json')
            weight = None
            if os.path.isfile(classes_weights_path):
                allContents = ""
                with open(classes_weights_path, "r", encoding='utf-8') as f:
                    for line in f:
                        allContents += line
                datasetInfo = json.loads(allContents)
                weight = np.array(datasetInfo['classWeight'])
            else:
                weight = common.calculate_weigths_labels(classes_weights_path, self.datasetLoader['train'], self.args.classNum)
                datasetInfo['classWeight'] = weight.tolist()
                print(datasetInfo)
                with open(classes_weights_path, "w") as f:
                    json.dump(datasetInfo, f)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.args.useCUDA).build_loss(mode=self.args.loss_type)
        if ("classNum" in datasetInfo):
            self.args.classNum = datasetInfo["classNum"]

        # Define Evaluator
        self.evaluator = common.Evaluator(self.args.classNum)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, self.args.epochs, len(self.datasetLoader['train']))

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            if self.args.useCUDA:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not self.args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            self.args.start_epoch = 0

    def training(self, epoch):
        self.model.train()
        batchNum = len(self.datasetLoader['train'])
        trainLoss = 0.0
        images, labels = None, None
        for i, sample in enumerate(self.datasetLoader['train']):
            images, labels = sample['image'], sample['label']
            labels = labels.squeeze(1) # 标签向量去除通道维度
            if self.args.useCUDA:
                images, labels = images.cuda(), labels.cuda()

            output = self.model(images)
            loss = self.criterion(output, labels)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            trainLoss += loss.item()
            print('\tTraining loss: %.3f' % (trainLoss / (i + 1)))
            self.writer.add_scalar('training/total_loss_iter', loss.item(), i + batchNum * epoch)

        print('[Epoch: %d]' % (epoch))
        print('Loss: %.3f' % trainLoss)
        self.writer.add_scalar('training/total_loss_epoch', trainLoss, epoch)
        self.writer.add_images("training/image", images, epoch)
        self.writer.add_images("training/labels", labels.unsqueeze(1) / self.args.classNum, epoch)
        self.writer.add_images("training/predicts", torch.argmax(output, dim = 1, keepdim=True) / self.args.classNum, epoch)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': self.model.module.state_dict(),
                                        'optimizer': self.optimizer.state_dict(),
                                        'best_pred': self.best_pred}, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        images, labels = None, None
        for i, sample in enumerate(self.datasetLoader['valid']):
            images, labels = sample['image'], sample['label']
            labels = labels.squeeze(1)
            if self.args.useCUDA:
                images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output = self.model(images)

            loss = self.criterion(output, labels)
            test_loss += loss.item()
            print('Validation loss: %.3f' % (test_loss / (i + 1)))
            labels = labels.cpu().numpy()
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(labels, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + images.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        self.writer.add_scalar('validation/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('validation/mIoU', mIoU, epoch)
        self.writer.add_scalar('validation/Acc', Acc, epoch)
        self.writer.add_scalar('validation/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('validation/fwIoU', FWIoU, epoch)
        self.writer.add_images("validation/image", images, epoch)
        self.writer.add_images("validation/labels", torch.Tensor(labels).unsqueeze(1) / self.args.classNum, epoch)
        self.writer.add_images("validation/predicts", torch.argmax(output, dim = 1, keepdim=True) / self.args.classNum, epoch)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,}, is_best)

if __name__ == "__main__":
   main()
