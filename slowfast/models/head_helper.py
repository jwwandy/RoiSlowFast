#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
# from detectron2.layers import ROIAlign
from torchvision.ops import RoIAlign

class RoiAlignComponent(nn.Module):
    def __init__(self, pool_size, resolution, scale_factor, pathway, aligned=True):
        super().__init__()
        self.roi_align = RoIAlign(
            resolution,
            spatial_scale=1.0 / scale_factor,
            sampling_ratio=0,
            aligned=aligned,
        )
        # self.add_module("s{}_roi".format(pathway), roi_align)
        
        self.temporal_pool = nn.AvgPool3d(pool_size, stride=1)
        # self.add_module("t{}_pool_roi".format(pathway), temporal_pool)

        self.spatial_pool = nn.MaxPool2d(resolution, stride=1)
        # self.add_module("s{}_pool_roi".format(pathway), spatial_pool)
        
        self.pathway = pathway
    
    def forward(self, input, bboxes, masks):
        B, C, T = input.shape[0:3]
        # B, C, T, H, W -> B, T, C, H, W
        input = input.permute((0, 2, 1, 3, 4))
        input = input.contiguous()
        # B*T, C, H, W
        input = input.view(-1, *input.shape[2:])
        # roi_align = getattr(self, "s{}_roi".format(self.pathway))
        # B*T*N, C, output_size[0], output_size[1]
        out = self.roi_align(input, bboxes)
        output_sizes = out.shape[2:4]

        # B*T*N, 1, 1, 1
        masks = masks.view(-1, 1, 1, 1)
        out = out * masks
        # B, T, N, C, output_size[0], output_size[1]
        out = out.view(B, T, len(masks)//(B*T), C, output_sizes[0], output_sizes[1])
        masks_denom = masks.view(B, T, -1).sum(2)
        masks_denom = masks_denom.view(B, T, 1, 1, 1)
        # B, T, C, output_size[0], output_size[1]
        out = out.sum(2) / (masks_denom + 1e-8)
        
        # Perform Temporal Pooling
        # B, T, C, output_size[0], output_size[1] ->  B, C, T, output_size[0], output_size[1]
        out = out.permute((0, 2, 1, 3, 4))
        # t_pool = getattr(self, "t{}_pool_roi".format(self.pathway))
        # B, C, 1, output_size[0], output_size[1]
        out_t = self.temporal_pool(out).squeeze(dim=2)
        
        # Perform Spatial Pooling
        # B, C, output_size[0], output_size[1]
        # s_pool = getattr(self, "s{}_pool_roi".format(self.pathway))
        # B, C, 1, 1, 1
        out_s = self.spatial_pool(out_t).unsqueeze(dim=2)
        return out_s


class ResNetBboxClassifierHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        roi_type,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        super(ResNetBboxClassifierHead, self).__init__()
        # assert (
        #     len({len(pool_size), len(dim_in)}) == 1
        # ), "pathway dimensions are not consistent."
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.num_pathways = len(pool_size)
        self.resolution = resolution

        for pathway in range(self.num_pathways):
            for pathway_roi_type in roi_type[pathway]:
                if pathway_roi_type == 0:
                    avg_pool = nn.AvgPool3d(pool_size[pathway][0])
                    self.add_module("original_avg_pool_{}".format(pathway), avg_pool)
                else:
                    roi_comp = RoiAlignComponent(pool_size[pathway][1], resolution[pathway], scale_factor[pathway], pathway, aligned=aligned)
                    self.add_module("roi_comp_{}".format(pathway), roi_comp)
        self.roi_type = roi_type
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

        if isinstance(num_classes, (list, tuple)):
            self.projection_verb = nn.Linear(sum(dim_in), num_classes[0], bias=True)
            self.projection_noun = nn.Linear(sum(dim_in), num_classes[1], bias=True)
        else:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

    def forward(self, inputs, bboxes, masks):
        '''
        bboxes: B*T*N, 5
        masks: B*T*N,
        '''
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        bboxes_i = 0
        for pathway in range(self.num_pathways):
            input = inputs[pathway]
            assert len(input.shape) == 5
            
            for pathway_roi_type in self.roi_type[pathway]:
                if pathway_roi_type == 0:
                    avg_pool = getattr(self, "original_avg_pool_{}".format(pathway))
                    input_normal = input.detach().clone()
                    out_s_normal = avg_pool(input_normal)
                else:
                    roi_comp = getattr(self, "roi_comp_{}".format(pathway))
                    input_normal = input.detach().clone()
                    out_s_normal = roi_comp(input_normal, bboxes[bboxes_i], masks[bboxes_i])
                    bboxes_i += 1

                pool_out.append(out_s_normal)

        # B x Cs x 1 x 1 x 1
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        if isinstance(self.num_classes, (list, tuple)):
            x_v = self.projection_verb(x)
            x_n = self.projection_noun(x)

            # Performs fully convlutional inference.
            if not self.training: # act here should be sigmoid
                x_v = self.act(x_v)
                x_v = x_v.mean([1, 2, 3])

            x_v = x_v.view(x_v.shape[0], -1)

            # Performs fully convlutional inference.
            if not self.training:
                x_n = self.act(x_n)
                x_n = x_n.mean([1, 2, 3])

            x_n = x_n.view(x_n.shape[0], -1)
            return (x_v, x_n)
        else:
            x = self.projection(x)

            # Performs fully convlutional inference.
            if not self.training:
                x = self.act(x)
                x = x.mean([1, 2, 3])

            x = x.view(x.shape[0], -1)
            return x

class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # x = x.view(x.shape[0], -1)
        # x = self.projection(x)
        # x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        feature_extraction=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.feature_extraction = feature_extraction
        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if isinstance(num_classes, (list, tuple)):
            self.projection_verb = nn.Linear(sum(dim_in), num_classes[0], bias=True)
            self.projection_noun = nn.Linear(sum(dim_in), num_classes[1], bias=True)
        else:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.num_classes = num_classes
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        x_feat = None
        if self.feature_extraction:
            x_feat = x.clone()
            if not self.training:
                x_feat = x_feat.mean([2, 3, 4])
            # assert x_feat.shape[2] == x_feat.shape[3] == x_feat.shape[4] == 1
            x_feat = x_feat.view(x.shape[0], -1).detach() #.cpu()

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if isinstance(self.num_classes, (list, tuple)):
            x_v = self.projection_verb(x)
            x_n = self.projection_noun(x)

            # Performs fully convlutional inference.
            if not self.training:
                x_v = self.act(x_v)
                x_v = x_v.mean([1, 2, 3])

            x_v = x_v.view(x_v.shape[0], -1)

            # Performs fully convlutional inference.
            if not self.training:
                x_n = self.act(x_n)
                x_n = x_n.mean([1, 2, 3])

            x_n = x_n.view(x_n.shape[0], -1)
            if self.feature_extraction:
                return (x_v, x_n), x_feat
            else:
                return (x_v, x_n)
        else:
            x = self.projection(x)

            # Performs fully convlutional inference.
            if not self.training:
                x = self.act(x)
                x = x.mean([1, 2, 3])

            x = x.view(x.shape[0], -1)
            if self.feature_extraction:
                return x, x_feat
            else:
                return x
