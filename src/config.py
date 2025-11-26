import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import datetime
import torch.nn as nn
import argparse
import time
import torchvision

class Config:
    def __init__(self, args):
        self.seed = 1029
        self.pass_channel = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.norm = False

        # File paths
        timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
        self.workdir = f"runs/{args.model}_{args.trainset}_{args.channel_type}_{timestamp}"
        self.samples = os.path.join(self.workdir, "samples")
        self.models = os.path.join(self.workdir, "models")
        self.log = os.path.join(self.workdir, "log.txt")

        # Training setup
        self.learning_rate = 1e-4
        # self.total_epochs = 1_000_000
        self.total_epochs = 20
        self.print_step = 100

        # Dataset setup
        if args.trainset == 'CIFAR10':
            self.image_dims = (3, 32, 32)
            self.batch_size = 128
            self.train_data_dir = "dataset/raw/CIFAR10/"
            self.test_data_dir = "dataset/raw/CIFAR10/"
            self.save_model_freq = 5

            self.channel_number = int(args.C)
            self.downsample = 2

            self.encoder_kwargs = dict(
                model=args.model,
                img_size=(self.image_dims[1], self.image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 256],
                depths=[2, 4],
                num_heads=[4, 8],
                C=self.channel_number,
                window_size=2,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            self.decoder_kwargs = dict(
                model=args.model,
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[256, 128],
                depths=[4, 2],
                num_heads=[8, 4],
                C=self.channel_number,
                window_size=2,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )

        elif args.trainset == 'div2k':
            self.image_dims = (3, 256, 256)
            self.batch_size = 16
            base_path = "dataset/raw"
            if args.testset == 'kodak':
                self.test_data_dir = ["dataset/raw/kodak"]
            elif args.testset == 'clic21':
                self.test_data_dir = ["dataset/raw/clic2021/test"]
            elif args.testset == 'ffhq':
                self.test_data_dir = ["dataset/raw/ffhq"]
            self.save_model_freq = 100

            self.train_data_dir = [
                # base_path + '/clic2021/train',
                # base_path + '/clic2021/valid',
                # base_path + '/clic2020',
                base_path + '/div2k/DIV2K_train_HR',
                base_path + '/div2k/DIV2K_valid_HR'
            ]

            if args.model == 'DJSCC':
                channel_number = int(args.C)

            if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
                channel_number = int(args.C)
            else:
                channel_number = None
            self.downsample = 4

            if args.model_size == 'small':
                self.encoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    patch_size=2,
                    in_chans=3,
                    embed_dims=[128, 192, 256, 320],
                    depths=[2, 2, 2, 2],
                    num_heads=[4, 6, 8, 10],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )
                self.decoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    embed_dims=[320, 256, 192, 128],
                    depths=[2, 2, 2, 2],
                    num_heads=[10, 8, 6, 4],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )
            elif args.model_size == 'base':
                self.encoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    patch_size=2,
                    in_chans=3,
                    embed_dims=[128, 192, 256, 320],
                    depths=[2, 2, 6, 2],
                    num_heads=[4, 6, 8, 10],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )
                self.decoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    embed_dims=[320, 256, 192, 128],
                    depths=[2, 6, 2, 2],
                    num_heads=[10, 8, 6, 4],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )
            elif args.model_size == 'large':
                self.encoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    patch_size=2,
                    in_chans=3,
                    embed_dims=[128, 192, 256, 320],
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 6, 8, 10],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )
                self.decoder_kwargs = dict(
                    model=args.model,
                    img_size=(self.image_dims[1], self.image_dims[2]),
                    embed_dims=[320, 256, 192, 128],
                    depths=[2, 18, 2, 2],
                    num_heads=[10, 8, 6, 4],
                    C=channel_number,
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                )

