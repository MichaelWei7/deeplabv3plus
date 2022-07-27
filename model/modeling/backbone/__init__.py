from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(args, backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained = args.pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained = args.pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained = args.pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained = args.pretrained)
    else:
        raise NotImplementedError
