from time import time
from setuptools import setup
import torch
import warnings
import itertools
import math
from collections import OrderedDict
from torch import nn, Tensor
from typing import Tuple, List, Dict

from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection.roi_heads import paste_masks_in_image
from .setup import ModelSetup

"""
This model is modified from the source code of torchvision.
(source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py)
"""


class MultimodalFuseModel(nn.Module):
    """

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(
        self,
        setup: ModelSetup,
        clinical_backbone=None,
    ):

        super(MultimodalFuseModel, self).__init__()
        self.setup = setup
        # self.fixation_backbone = fixation_backbone
        self.fusing_channels = self.setup.backbone_out_channels
        # used only on torchscript mode
        self._has_warned = False

        self.clinical_convs = clinical_backbone
        self.setup = setup
        self.feature_keys = ["0"]
        # example_img_features = self.backbone(
        #     self.transform([torch.ones(3, 2048, 2048)])[0].tensors
        # )
        #
        # # if isinstance(example_img_features, OrderedDict):
        # if isinstance(example_img_features, torch.Tensor):
        #     self.feature_keys = ["0"]
        #     example_img_features = OrderedDict([("0", example_img_features)])
        # else:
        #     self.feature_keys = example_img_features.keys()
        #
        # last_key = list(example_img_features.keys())[-1]
        # self.image_feature_map_size = example_img_features[last_key].shape[-1]

        if self.setup.use_clinical:
            self.build_clinical_model()

    def build_clinical_model(self,):
        clincial_cat_is_used = self.setup.has_categorical_clinical_features()
        if clincial_cat_is_used:
            self.gender_emb_layer = nn.Embedding(
                2,
                self.setup.get_emb_dim(),
            )
        # spatialise module
        if self.setup.spatialise_clinical:
            if self.setup.spatialise_method == "convs":
                expand_times = math.log((self.setup.image_size/16), 2)

                assert (
                    expand_times.is_integer(),
                    f"The expand_times should be interger but found {expand_times}",
                )

                expand_conv_modules = list(
                    itertools.chain.from_iterable(
                        [
                            [
                                nn.ConvTranspose2d(
                                    (
                                        self.setup.get_input_dim_for_spa()
                                        if i == 0
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                    (self.setup.clinical_expand_conv_channels),
                                    kernel_size=2,
                                    stride=2,
                                ),
                                nn.BatchNorm2d(
                                    (self.setup.clinical_expand_conv_channels),
                                ),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(
                                    (self.setup.clinical_expand_conv_channels),
                                    (
                                        512
                                        if i == int(expand_times) - 1
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                ),
                                nn.BatchNorm2d(
                                    (
                                        512
                                        if i == int(expand_times) - 1
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                ),
                            ]
                            + (
                                []
                                if i == expand_times - 1
                                else [nn.ReLU(inplace=False),]
                            )
                            for i in range(
                                int(expand_times)
                            )  # expand to the same as image size.
                        ]
                    )
                )

                self.clinical_expand_conv = nn.Sequential(*expand_conv_modules,)
            elif self.setup.spatialise_method == "repeat":
                self.before_repeat = nn.ModuleDict({})
                for k in self.feature_keys:
                    self.before_repeat[k] = nn.Linear(
                        self.setup.clinical_input_channels, self.fusing_channels
                    )
            else:
                raise Exception(
                    f"Unsupported spatialisation method: {self.setup.spatialise_method}"
                )

            self.pre_spa = None

            if (
                not self.setup.pre_spatialised_layer is None
                and self.setup.pre_spatialised_layer > 0
            ):
                pre_spa_layers = itertools.chain.from_iterable(
                    [
                        [
                            nn.Linear(
                                (
                                    self.setup.get_input_dim_for_pre_spa()
                                    if i == 0
                                    else self.setup.clinical_input_channels
                                ),
                                self.setup.clinical_input_channels,
                            ),
                            nn.BatchNorm1d(self.setup.clinical_input_channels),
                            nn.LeakyReLU(),
                        ]
                        for i in range(self.setup.pre_spatialised_layer)
                    ]
                )
                self.pre_spa = nn.Sequential(*pre_spa_layers)
                self.pre_spa.append(
                    nn.Linear(
                        self.setup.clinical_input_channels,
                        self.setup.clinical_input_channels,
                    )
                )

            self._build_normal_fuse_convs()


    def _build_normal_fuse_convs(self,):
        if self.setup.fuse_depth == 0:
            return

        fuse_convs_modules = list(
            itertools.chain.from_iterable(
                [
                    [
                        nn.Conv2d(
                            (
                                self.get_fuse_input_channel()
                                if i == 0
                                else self.setup.clinical_conv_channels
                            ),
                            (
                                self.fusing_channels
                                if i == (self.setup.fuse_depth - 1)
                                else self.setup.clinical_conv_channels
                            ),
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(
                            (
                                self.fusing_channels
                                if i == (self.setup.fuse_depth - 1)
                                else self.setup.clinical_conv_channels
                            )
                        ),
                        nn.ReLU(inplace=False),
                    ]
                    for i in range(self.setup.fuse_depth)
                ]
            )
        )

        self.fuse_convs = nn.Sequential(*fuse_convs_modules)

    def get_fuse_input_channel(self,):
        if self.setup.fusion_strategy == "concat":
            return self.fusing_channels * 2
        elif self.setup.fusion_strategy == "add":
            return self.fusing_channels
        else:
            raise Exception(
                f"Unsupported fusion strategy: {self.setup.fusion_strategy}"
            )

    def get_clinical_features(self, clinical_num, clinical_cat, img_features):

        clinical_input = None
        if self.setup.use_clinical:
            if (
                self.setup.has_categorical_clinical_features()
                and self.setup.has_numerical_clinical_features()
            ):
                clincal_embout = self.gender_emb_layer(
                    torch.concat(clinical_cat, axis=0)
                )
                clinical_input = torch.concat(
                    [torch.stack(clinical_num, dim=0), clincal_embout], axis=1
                )
            elif self.setup.has_categorical_clinical_features():
                clinical_input = self.gender_emb_layer(
                    torch.concat(clinical_cat, axis=0)
                )
            elif self.setup.has_numerical_clinical_features():
                #clinical_input = torch.stack(clinical_num, dim=0)
                clinical_input = clinical_num
            else:
                raise ValueError("No clinical feature provided")

        if self.pre_spa:
            clinical_input = self.pre_spa(clinical_input)

        clinical_features = None
        if self.setup.spatialise_clinical:
            clinical_features = OrderedDict({})
            if self.setup.spatialise_method == "convs":
                clinical_expanded_input = self.clinical_expand_conv(
                    clinical_input[:, :, None, None]
                )
                self.last_clinical_expanded_input = clinical_expanded_input
                ##############
                clinical_expanded_input = torch.unsqueeze(clinical_expanded_input,dim=2)
                ################
                clinical_features = clinical_expanded_input.repeat(1,1,self.setup.image_size//16,1,1)
                if isinstance(clinical_features, torch.Tensor):
                    clinical_features = OrderedDict([("0", clinical_features)])
            elif self.setup.spatialise_method == "repeat":
                for k in self.feature_keys:
                    clinical_features[k] = self.before_repeat[k](clinical_input)[
                        :, :, None, None
                    ].repeat(
                        1, 1, img_features[k].shape[-2], img_features[k].shape[-1],
                    )
            else:
                raise Exception(
                    "Unsupported spatialise method: {self.setup.sptailise_method}"
                )

        return clinical_input, clinical_features

    def fuse_feature_maps(
        self, img_feature: torch.Tensor, clinical_feature: torch.Tensor
    ) -> torch.Tensor:
        if self.setup.fusion_strategy == "concat":
            return torch.concat([img_feature, clinical_feature], axis=1)
        elif self.setup.fusion_strategy == "add":
            return img_feature + clinical_feature
        else:
            raise Exception(
                f"Unsupported fusion strategy: {self.setup.fusion_strategyn}"
            )

    def fuse_features(self, img_features, clinical_features):
        features = OrderedDict({})
        k = "0"
        if self.setup.fusion_strategy == "add" and self.setup.fuse_depth == 0:
            features[k] = self.fuse_feature_maps(
                img_features[0], clinical_features[k]
            )

        else:
            features[k] = self.fuse_convs(
                self.fuse_feature_maps(img_features[k], clinical_features[k])
            )

        if self.setup.fusion_strategy == "add" and self.setup.fusion_residule:
            features[k] = features[k] + img_features[k] + clinical_features[k]

        return features

    def forward(self, images, clinical_num=None, clinical_cat=None, targets=None):

        """
        Args
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.setup.use_clinical:
            assert (not clinical_num is None) or (
                not clinical_cat is None
            ), "You're using clinical data, but they're not passed into model."

        # if self.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")


        img_features = images

        if isinstance(img_features, torch.Tensor):
            img_features = OrderedDict([("0", img_features)])

        clinical_input = None
        if self.setup.use_clinical:
            clinical_input, clinical_features = self.get_clinical_features(
                clinical_num, clinical_cat, img_features
            )

            if self.setup.spatialise_clinical:
                features = self.fuse_features(img_features, clinical_features)
            else:
                features = img_features
        else:
            features = img_features

        losses = {}
        return losses, features["0"]



