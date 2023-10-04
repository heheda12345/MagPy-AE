# trident resnet in mmdetection, graph break due to numpy

import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
# from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

import torch
from torch import nn as nn
from typing import Dict, Union, Tuple, Optional, List

import math
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler
from typing import Iterable, Optional

from mmcv.runner.dist_utils import master_only
from mmcv.utils.logging import get_logger, logger_initialized, print_log


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self) -> None:
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info: defaultdict = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from ..cnn import initialize
        from ..cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name: str) -> None:
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f'\n{name} - {param.shape}: '
                    f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s

class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class GeneralizedAttention(nn.Module):
    """GeneralizedAttention module.

    See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
    (https://arxiv.org/abs/1711.07971) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        spatial_range (int): The spatial range. -1 indicates no spatial range
            constraint. Default: -1.
        num_heads (int): The head number of empirical_attention module.
            Default: 9.
        position_embedding_dim (int): The position embedding dimension.
            Default: -1.
        position_magnitude (int): A multiplier acting on coord difference.
            Default: 1.
        kv_stride (int): The feature stride acting on key/value feature map.
            Default: 2.
        q_stride (int): The feature stride acting on query feature map.
            Default: 1.
        attention_type (str): A binary indicator string for indicating which
            items in generalized empirical_attention module are used.
            Default: '1111'.

            - '1000' indicates 'query and key content' (appr - appr) item,
            - '0100' indicates 'query content and relative position'
              (appr - position) item,
            - '0010' indicates 'key content only' (bias - appr) item,
            - '0001' indicates 'relative position only' (bias - position) item.
    """

    _abbr_ = 'gen_attention_block'

    def __init__(self,
                 in_channels: int,
                 spatial_range: int = -1,
                 num_heads: int = 9,
                 position_embedding_dim: int = -1,
                 position_magnitude: int = 1,
                 kv_stride: int = 2,
                 q_stride: int = 1,
                 attention_type: str = '1111'):

        super().__init__()

        # hard range means local range for non-local operation
        self.position_embedding_dim = (
            position_embedding_dim
            if position_embedding_dim > 0 else in_channels)

        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.spatial_range = spatial_range
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.attention_type = [bool(int(_)) for _ in attention_type]
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        if self.attention_type[0] or self.attention_type[1]:
            self.query_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
            self.query_conv.kaiming_init = True

        if self.attention_type[0] or self.attention_type[2]:
            self.key_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
            self.key_conv.kaiming_init = True

        self.v_dim = in_channels // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False)
        self.value_conv.kaiming_init = True

        if self.attention_type[1] or self.attention_type[3]:
            self.appr_geom_fc_x = nn.Linear(
                self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_x.kaiming_init = True

            self.appr_geom_fc_y = nn.Linear(
                self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_y.kaiming_init = True

        if self.attention_type[2]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias = nn.Parameter(appr_bias_value)

        if self.attention_type[3]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            geom_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.geom_bias = nn.Parameter(geom_bias_value)

        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_channels,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.spatial_range >= 0:
            # only works when non local is after 3*3 conv
            if in_channels == 256:
                max_len = 84
            elif in_channels == 512:
                max_len = 42

            max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
            local_constraint_map = np.ones(
                (max_len, max_len, max_len_kv, max_len_kv), dtype=int)
            for iy in range(max_len):
                for ix in range(max_len):
                    local_constraint_map[
                        iy, ix,
                        max((iy - self.spatial_range) //
                            self.kv_stride, 0):min((iy + self.spatial_range +
                                                    1) // self.kv_stride +
                                                   1, max_len),
                        max((ix - self.spatial_range) //
                            self.kv_stride, 0):min((ix + self.spatial_range +
                                                    1) // self.kv_stride +
                                                   1, max_len)] = 0

            self.local_constraint_map = nn.Parameter(
                torch.from_numpy(local_constraint_map).byte(),
                requires_grad=False)

        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None

        self.init_weights()

    def get_position_embedding(self,
                               h,
                               w,
                               h_kv,
                               w_kv,
                               q_stride,
                               kv_stride,
                               device,
                               dtype,
                               feat_dim,
                               wave_length=1000):
        # the default type of Tensor is float32, leading to type mismatch
        # in fp16 mode. Cast it to support fp16 mode.
        h_idxs = torch.linspace(0, h - 1, h).to(device=device, dtype=dtype)
        h_idxs = h_idxs.view((h, 1)) * q_stride

        w_idxs = torch.linspace(0, w - 1, w).to(device=device, dtype=dtype)
        w_idxs = w_idxs.view((w, 1)) * q_stride

        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).to(
            device=device, dtype=dtype)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride

        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).to(
            device=device, dtype=dtype)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride

        # (h, h_kv, 1)
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude

        # (w, w_kv, 1)
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude

        feat_range = torch.arange(0, feat_dim / 4).to(
            device=device, dtype=dtype)

        dim_mat = torch.Tensor([wave_length]).to(device=device, dtype=dtype)
        dim_mat = dim_mat**((4. / feat_dim) * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))

        embedding_x = torch.cat(
            ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

        embedding_y = torch.cat(
            ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

        return embedding_x, embedding_y

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        num_heads = self.num_heads

        # use empirical_attention
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape

        if self.attention_type[0] or self.attention_type[1]:
            proj_query = self.query_conv(x_q).view(
                (n, num_heads, self.qk_embed_dim, h * w))
            proj_query = proj_query.permute(0, 1, 3, 2)

        if self.attention_type[0] or self.attention_type[2]:
            proj_key = self.key_conv(x_kv).view(
                (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        if self.attention_type[1] or self.attention_type[3]:
            position_embed_x, position_embed_y = self.get_position_embedding(
                h, w, h_kv, w_kv, self.q_stride, self.kv_stride,
                x_input.device, x_input.dtype, self.position_embedding_dim)
            # (n, num_heads, w, w_kv, dim)
            position_feat_x = self.appr_geom_fc_x(position_embed_x).\
                view(1, w, w_kv, num_heads, self.qk_embed_dim).\
                permute(0, 3, 1, 2, 4).\
                repeat(n, 1, 1, 1, 1)

            # (n, num_heads, h, h_kv, dim)
            position_feat_y = self.appr_geom_fc_y(position_embed_y).\
                view(1, h, h_kv, num_heads, self.qk_embed_dim).\
                permute(0, 3, 1, 2, 4).\
                repeat(n, 1, 1, 1, 1)

            position_feat_x /= math.sqrt(2)
            position_feat_y /= math.sqrt(2)

        # accelerate for saliency only
        if (np.sum(self.attention_type) == 1) and self.attention_type[2]:
            appr_bias = self.appr_bias.\
                view(1, num_heads, 1, self.qk_embed_dim).\
                repeat(n, 1, 1, 1)

            energy = torch.matmul(appr_bias, proj_key).\
                view(n, num_heads, 1, h_kv * w_kv)

            h = 1
            w = 1
        else:
            # (n, num_heads, h*w, h_kv*w_kv), query before key, 540mb for
            if not self.attention_type[0]:
                energy = torch.zeros(
                    n,
                    num_heads,
                    h,
                    w,
                    h_kv,
                    w_kv,
                    dtype=x_input.dtype,
                    device=x_input.device)

            # attention_type[0]: appr - appr
            # attention_type[1]: appr - position
            # attention_type[2]: bias - appr
            # attention_type[3]: bias - position
            if self.attention_type[0] or self.attention_type[2]:
                if self.attention_type[0] and self.attention_type[2]:
                    appr_bias = self.appr_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim)
                    energy = torch.matmul(proj_query + appr_bias, proj_key).\
                        view(n, num_heads, h, w, h_kv, w_kv)

                elif self.attention_type[0]:
                    energy = torch.matmul(proj_query, proj_key).\
                        view(n, num_heads, h, w, h_kv, w_kv)

                elif self.attention_type[2]:
                    appr_bias = self.appr_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim).\
                        repeat(n, 1, 1, 1)

                    energy += torch.matmul(appr_bias, proj_key).\
                        view(n, num_heads, 1, 1, h_kv, w_kv)

            if self.attention_type[1] or self.attention_type[3]:
                if self.attention_type[1] and self.attention_type[3]:
                    geom_bias = self.geom_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim)

                    proj_query_reshape = (proj_query + geom_bias).\
                        view(n, num_heads, h, w, self.qk_embed_dim)

                    energy_x = torch.matmul(
                        proj_query_reshape.permute(0, 1, 3, 2, 4),
                        position_feat_x.permute(0, 1, 2, 4, 3))
                    energy_x = energy_x.\
                        permute(0, 1, 3, 2, 4).unsqueeze(4)

                    energy_y = torch.matmul(
                        proj_query_reshape,
                        position_feat_y.permute(0, 1, 2, 4, 3))
                    energy_y = energy_y.unsqueeze(5)

                    energy += energy_x + energy_y

                elif self.attention_type[1]:
                    proj_query_reshape = proj_query.\
                        view(n, num_heads, h, w, self.qk_embed_dim)
                    proj_query_reshape = proj_query_reshape.\
                        permute(0, 1, 3, 2, 4)
                    position_feat_x_reshape = position_feat_x.\
                        permute(0, 1, 2, 4, 3)
                    position_feat_y_reshape = position_feat_y.\
                        permute(0, 1, 2, 4, 3)

                    energy_x = torch.matmul(proj_query_reshape,
                                            position_feat_x_reshape)
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)

                    energy_y = torch.matmul(proj_query_reshape,
                                            position_feat_y_reshape)
                    energy_y = energy_y.unsqueeze(5)

                    energy += energy_x + energy_y

                elif self.attention_type[3]:
                    geom_bias = self.geom_bias.\
                        view(1, num_heads, self.qk_embed_dim, 1).\
                        repeat(n, 1, 1, 1)

                    position_feat_x_reshape = position_feat_x.\
                        view(n, num_heads, w * w_kv, self.qk_embed_dim)

                    position_feat_y_reshape = position_feat_y.\
                        view(n, num_heads, h * h_kv, self.qk_embed_dim)

                    energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
                    energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)

                    energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
                    energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)

                    energy += energy_x + energy_y

            energy = energy.view(n, num_heads, h * w, h_kv * w_kv)

        if self.spatial_range >= 0:
            cur_local_constraint_map = \
                self.local_constraint_map[:h, :w, :h_kv, :w_kv].\
                contiguous().\
                view(1, 1, h*w, h_kv*w_kv)

            energy = energy.masked_fill_(cur_local_constraint_map,
                                         float('-inf'))

        attention = F.softmax(energy, 3)

        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, h, w)

        out = self.proj_conv(out)

        # output is downsampled, upsample back to input size
        if self.q_downsample is not None:
            out = F.interpolate(
                out,
                size=x_input.shape[2:],
                mode='bilinear',
                align_corners=False)

        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)

class TridentConv(BaseModule):
    """Trident Convolution Module.

    Args:
        in_channels (int): Number of channels in input.
        out_channels (int): Number of channels in output.
        kernel_size (int): Size of convolution kernel.
        stride (int, optional): Convolution stride. Default: 1.
        trident_dilations (tuple[int, int, int], optional): Dilations of
            different trident branch. Default: (1, 2, 3).
        test_branch_idx (int, optional): In inference, all 3 branches will
            be used if `test_branch_idx==-1`, otherwise only branch with
            index `test_branch_idx` will be used. Default: 1.
        bias (bool, optional): Whether to use bias in convolution or not.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 trident_dilations=(1, 2, 3),
                 test_branch_idx=1,
                 bias=False,
                 init_cfg=None):
        super(TridentConv, self).__init__(init_cfg)
        self.num_branch = len(trident_dilations)
        self.with_bias = bias
        self.test_branch_idx = test_branch_idx
        self.stride = _pair(stride)
        self.kernel_size = _pair(kernel_size)
        self.paddings = _pair(trident_dilations)
        self.dilations = trident_dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def extra_repr(self):
        tmpstr = f'in_channels={self.in_channels}'
        tmpstr += f', out_channels={self.out_channels}'
        tmpstr += f', kernel_size={self.kernel_size}'
        tmpstr += f', num_branch={self.num_branch}'
        tmpstr += f', test_branch_idx={self.test_branch_idx}'
        tmpstr += f', stride={self.stride}'
        tmpstr += f', paddings={self.paddings}'
        tmpstr += f', dilations={self.dilations}'
        tmpstr += f', bias={self.bias}'
        return tmpstr

    def forward(self, inputs):
        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, self.stride, padding,
                         dilation) for input, dilation, padding in zip(
                             inputs, self.dilations, self.paddings)
            ]
        else:
            assert len(inputs) == 1
            outputs = [
                F.conv2d(inputs[0], self.weight, self.bias, self.stride,
                         self.paddings[self.test_branch_idx],
                         self.dilations[self.test_branch_idx])
            ]

        return outputs



# Since TridentNet is defined over ResNet50 and ResNet101, here we
# only support TridentBottleneckBlock.
class TridentBottleneck(Bottleneck):
    """BottleBlock for TridentResNet.

    Args:
        trident_dilations (tuple[int, int, int]): Dilations of different
            trident branch.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        concat_output (bool): Whether to concat the output list to a Tensor.
            `True` only in the last Block.
    """

    def __init__(self, trident_dilations, test_branch_idx, concat_output,
                 **kwargs):

        super(TridentBottleneck, self).__init__(**kwargs)
        self.trident_dilations = trident_dilations
        self.num_branch = len(trident_dilations)
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        self.conv2 = TridentConv(
            self.planes,
            self.planes,
            kernel_size=3,
            stride=self.conv2_stride,
            bias=False,
            trident_dilations=self.trident_dilations,
            test_branch_idx=test_branch_idx,
            init_cfg=dict(
                type='Kaiming',
                distribution='uniform',
                mode='fan_in',
                override=dict(name='conv2')))

    def forward(self, x):

        def _inner_forward(x):
            num_branch = (
                self.num_branch
                if self.training or self.test_branch_idx == -1 else 1)
            identity = x
            if not isinstance(x, list):
                x = (x, ) * num_branch
                identity = x
                if self.downsample is not None:
                    identity = [self.downsample(b) for b in x]

            out = [self.conv1(b) for b in x]
            out = [self.norm1(b) for b in out]
            out = [self.relu(b) for b in out]

            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = [self.norm2(b) for b in out]
            out = [self.relu(b) for b in out]
            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv2_plugin_names)

            out = [self.conv3(b) for b in out]
            out = [self.norm3(b) for b in out]

            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv3_plugin_names)

            out = [
                out_b + identity_b for out_b, identity_b in zip(out, identity)
            ]
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = [self.relu(b) for b in out]
        if self.concat_output:
            out = torch.cat(out, dim=0)
        return out

def make_trident_res_layer(block,
                           inplanes,
                           planes,
                           num_blocks,
                           stride=1,
                           trident_dilations=(1, 2, 3),
                           style='pytorch',
                           with_cp=False,
                           conv_cfg=None,
                           norm_cfg=dict(type='BN'),
                           dcn=None,
                           plugins=None,
                           test_branch_idx=-1):
    """Build Trident Res Layers."""

    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        downsample.extend([
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=conv_stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1]
        ])
        downsample = nn.Sequential(*downsample)

    layers = []
    for i in range(num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride if i == 0 else 1,
                trident_dilations=trident_dilations,
                downsample=downsample if i == 0 else None,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=plugins,
                test_branch_idx=test_branch_idx,
                concat_output=True if i == num_blocks - 1 else False))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


class TridentResNet(ResNet):
    """The stem layer, stage 1 and stage 2 in Trident ResNet are identical to
    ResNet, while in stage 3, Trident BottleBlock is utilized to replace the
    normal BottleBlock to yield trident output. Different branch shares the
    convolution weight but uses different dilations to achieve multi-scale
    output.

                               / stage3(b0) \
    x - stem - stage1 - stage2 - stage3(b1) - output
                               \ stage3(b2) /

    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        trident_dilations (tuple[int]): Dilations of different trident branch.
            len(trident_dilations) should be equal to num_branch.
    """  # noqa

    def __init__(self, depth, num_branch, test_branch_idx, trident_dilations,
                 **kwargs):

        assert num_branch == len(trident_dilations)
        assert depth in (50, 101, 152)
        super(TridentResNet, self).__init__(depth, **kwargs)
        assert self.num_stages == 3
        self.test_branch_idx = test_branch_idx
        self.num_branch = num_branch

        last_stage_idx = self.num_stages - 1
        stride = self.strides[last_stage_idx]
        dilation = trident_dilations
        dcn = self.dcn if self.stage_with_dcn[last_stage_idx] else None
        if self.plugins is not None:
            stage_plugins = self.make_stage_plugins(self.plugins,
                                                    last_stage_idx)
        else:
            stage_plugins = None
        planes = self.base_channels * 2**last_stage_idx
        res_layer = make_trident_res_layer(
            TridentBottleneck,
            inplanes=(self.block.expansion * self.base_channels *
                      2**(last_stage_idx - 1)),
            planes=planes,
            num_blocks=self.stage_blocks[last_stage_idx],
            stride=stride,
            trident_dilations=dilation,
            style=self.style,
            with_cp=self.with_cp,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=dcn,
            plugins=stage_plugins,
            test_branch_idx=self.test_branch_idx)

        layer_name = f'layer{last_stage_idx + 1}'

        self.__setattr__(layer_name, res_layer)
        self.res_layers.pop(last_stage_idx)
        self.res_layers.insert(last_stage_idx, layer_name)

        self._freeze_stages()
# '''


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