# trident resnet in mmdetection, graph break due to numpy
from mmdet.models.backbones import TridentResNet
import torch

def get_model():
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            position='after_conv2')
    ]

    tridentresnet_config = dict(
        num_branch=3,
        test_branch_idx=1,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        trident_dilations=(1, 2, 3),
        out_indices=(2, ),
        plugins=plugins,
    )

    model = TridentResNet(50, num_stages=3, **tridentresnet_config).cuda()
    return model


def get_input(batch_size):
    return (torch.randn((batch_size, 3, 224, 224)).cuda(),), {}