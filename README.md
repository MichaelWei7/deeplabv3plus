# DeepLab V3 plus

原项目地址: https://github.com/jfzhang95/pytorch-deeplab-xception

其中:

`model/customize_service.py` 用于 ModelArts 服务部署.

`model/config.json` 用于配置 ModelArts 服务部署.

`train-workflow.py` 用于 ModelArts 模型训练.

`pip-requirements.txt` 用于 ModelArts 的训练环境搭建.

train_coco.sh: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset coco
```

train_voc.sh

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.007 --workers 4 --use-sbd True --epochs 50 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
```