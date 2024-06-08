from functools import partial
import itertools
import math
import numpy as np
import torch.nn as nn
import torch
from scipy.special import softmax
from pytorch_tabnet.utils import check_embedding_parameters
from unet3d.models.pytorch.unet_fuse import MultimodalFuseModel
from unet3d.models.pytorch.classification.decoder import MyronenkoDecoder, MirroredDecoder
from unet3d.models.pytorch.classification.myronenko import MyronenkoEncoder, MyronenkoConvolutionBlock
from unet3d.models.pytorch.classification.resnet import conv1x1x1
from unet3d.models.pytorch.setup import ModelSetup
from unet3d.models.pytorch.tab_network import TabNet
from pytorch_tabnet.utils import create_group_matrix
from pytorch_tabnet.augmentations import ClassificationSMOTE

class VariationalBlock(nn.Module):
    def __init__(self, in_size, n_features, out_size, return_parameters=False):
        super(VariationalBlock, self).__init__()
        self.n_features = n_features
        self.return_parameters = return_parameters
        self.dense1 = nn.Linear(in_size, out_features=n_features*2)
        self.dense2 = nn.Linear(self.n_features, out_size)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.dense1(x)
        mu, logvar = torch.split(x, self.n_features, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.dense2(z)
        if self.return_parameters:
            return out, mu, logvar, z
        else:
            return out, mu, logvar

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=[], decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                 encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                 decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.base_width = base_width
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder_class(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size)
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                                decoder_blocks)
        self.decoder = decoder_class(base_width=base_width, layer_blocks=decoder_blocks,
                                     upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)
        self.set_final_convolution(n_features)
        self.set_activation(activation=activation)

    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            if decoder_class is None:
                decoder_class = MirroredDecoder
        elif decoder_blocks is None:
            decoder_blocks = [1] * len(encoder_blocks)
            if decoder_class is None:
                decoder_class = MyronenkoDecoder
        return decoder_class, decoder_blocks

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
class ConvolutionalAutoEncoder_tab(nn.Module):
    def __init__(self, input_dim=38,output_dim=8, n_features=1,cat_idx=[],cat_dim=[], base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                 encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                 decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder_tab, self).__init__()
        self.base_width = base_width
        self.model_setup = ModelSetup(including_clinical_cat=[])
        if self.model_setup.use_clinical:
            self.group_matrix = create_group_matrix([], input_dim)
            cat_emb_dim = 2
            updated_params = check_embedding_parameters(cat_dim,
                                                        cat_idx,
                                                        cat_emb_dim)
            cat_dim, cat_idx, cat_emb_dim = updated_params
            self.tablenet = TabNet(input_dim, output_dim,cat_idxs=cat_idx,cat_dims=cat_dim,cat_emb_dim=cat_emb_dim,group_attention_matrix=self.group_matrix)
            self.fuse_model =  MultimodalFuseModel(self.model_setup,self.group_matrix)
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder_class(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size)
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                                decoder_blocks)
        self.decoder = decoder_class(base_width=base_width, layer_blocks=decoder_blocks,
                                     upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)
        self.set_final_convolution(n_features)
        self.set_activation(activation=activation)
        self.cls = True
        self.final_mapping = nn.Linear(32768, output_dim, bias=False)
        self.final_mapping2 = nn.Linear(262144, output_dim, bias=False)
        self.batch = torch.nn.BatchNorm1d(num_features=3)
        self.relu = torch.nn.ReLU()
        self.aug = ClassificationSMOTE(p=0.2)
        self.activation_cls = nn.Softmax(dim=1)
    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            if decoder_class is None:
                decoder_class = MirroredDecoder
        elif decoder_blocks is None:
            decoder_blocks = [1] * len(encoder_blocks)
            if decoder_class is None:
                decoder_class = MyronenkoDecoder
        return decoder_class, decoder_blocks


    def forward(self, x, tab_info,tab_y):

        tab_info,tab_y = self.aug(tab_info,tab_y)
        x = self.encoder(x)
        seg_x = self.decoder(x)
        seg_x = self.final_convolution(seg_x)
        if self.activation is not None:
            seg_x = self.activation(seg_x)

        table_feature, _ = self.tablenet(tab_info)

        #3d_fuse
        # _,fuse_x = self.fuse_model(images=x,clinical_num=table_feature)
        # cls_x = fuse_x
        # del table_feature
        # cls_x = torch.flatten(cls_x,1,-1)
        # cls_x = self.final_mapping(cls_x)

        #1d_fuse
        #fl_seg_x = torch.flatten(seg_x_sq,1,-1)
        #fl_seg_x = self.final_mapping2(fl_seg_x)
        # cls_x = fl_seg_x + table_feature
        # cls_x = self.relu(cls_x)

        cls_x = table_feature
        cls_x = self.activation_cls(cls_x)
        return seg_x,cls_x,tab_info,tab_y


class MyronenkoVariationalLayer(nn.Module):
    def __init__(self, in_features, input_shape, reduced_features=16, latent_features=128,
                 conv_block=MyronenkoConvolutionBlock, conv_stride=2, upsampling_mode="trilinear",
                 align_corners_upsampling=False):
        super(MyronenkoVariationalLayer, self).__init__()
        self.in_conv = conv_block(in_planes=in_features, planes=reduced_features, stride=conv_stride)
        self.reduced_shape = tuple(np.asarray((reduced_features, *np.divide(input_shape, conv_stride)), dtype=np.int))
        self.in_size = np.prod(self.reduced_shape, dtype=np.int)
        self.var_block = VariationalBlock(in_size=self.in_size, out_size=self.in_size, n_features=latent_features)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = conv1x1x1(in_planes=reduced_features, out_planes=in_features, stride=1)
        self.upsample = partial(nn.functional.interpolate, scale_factor=conv_stride, mode=upsampling_mode,
                                align_corners=align_corners_upsampling)

    def forward(self, x):
        x = self.in_conv(x).flatten(start_dim=1)
        x, mu, logvar = self.var_block(x)
        x = self.relu(x).view(-1, *self.reduced_shape)
        x = self.out_conv(x)
        x = self.upsample(x)
        return x, mu, logvar


class VariationalAutoEncoder(ConvolutionalAutoEncoder):
    def __init__(self, n_reduced_latent_feature_maps=16, vae_features=128, variational_layer=MyronenkoVariationalLayer,
                 input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear", encoder_class=MyronenkoEncoder,
                 decoder_class=MyronenkoDecoder, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False,
                 activation=None, use_transposed_convolutions=False, var_layer_stride=2):
        super(VariationalAutoEncoder, self).__init__(input_shape=input_shape, n_features=n_features,
                                                     base_width=base_width, encoder_blocks=encoder_blocks,
                                                     decoder_blocks=decoder_blocks, feature_dilation=feature_dilation,
                                                     downsampling_stride=downsampling_stride,
                                                     interpolation_mode=interpolation_mode, encoder_class=encoder_class,
                                                     decoder_class=decoder_class, n_outputs=n_outputs, layer_widths=layer_widths,
                                                     decoder_mirrors_encoder=decoder_mirrors_encoder,
                                                     activation=activation,
                                                     use_transposed_convolutions=use_transposed_convolutions)
        if vae_features is not None:
            depth = len(encoder_blocks) - 1
            n_latent_feature_maps = base_width * (feature_dilation ** depth)
            latent_image_shape = np.divide(input_shape, downsampling_stride ** depth)
            self.var_layer = variational_layer(in_features=n_latent_feature_maps,
                                               input_shape=latent_image_shape,
                                               reduced_features=n_reduced_latent_feature_maps,
                                               latent_features=vae_features,
                                               upsampling_mode=interpolation_mode,
                                               conv_stride=var_layer_stride)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar

    def test(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(mu)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar


class LabeledVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self, *args, n_outputs=None, base_width=32, **kwargs):
        super().__init__(*args, n_outputs=n_outputs, base_width=base_width, **kwargs)
        self.final_convolution = conv1x1x1(in_planes=base_width, out_planes=n_outputs, stride=1)
