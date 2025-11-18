import torch.optim as optim
from models.SwinJSCC.network import *
from data.datasets import get_loader
from utils.helpers import *
from utils.logger import *
from config import Config
from pathlib import Path
from datetime import datetime
from training.loss import *
from training.trainer import Trainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import argparse
import time
import torchvision

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwinJSCC")
    p.add_argument("--training", action="store_true", help="training mode")
    p.add_argument("--trainset", type=str, default="div2k",
                choices=["CIFAR10", "div2k"])
    p.add_argument("--testset", type=str, default="kodak",
                choices=["kodak", "clic21", "ffhq"])
    p.add_argument("--distortion-metric", type=str, default="MSE",
                choices=["MSE", "MS-SSIM"])
    p.add_argument("--model", type=str,
                default="SwinJSCC_w/_SAandRA",
                choices=["SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA",
                            "SwinJSCC_w/_RA", "SwinJSCC_w/_SAandRA"])
    p.add_argument("--channel-type", type=str, default="awgn",
                choices=["awgn", "rayleigh"])
    p.add_argument("--C", type=str, default="96", help="bottleneck dimension(s), comma separated")
    p.add_argument("--multiple-snr", type=str, default="10",
                help="SNR values for test, comma separated")
    p.add_argument("--model_size", type=str, default="base",
                choices=["small", "base", "large"])
    return p

def load_weights(model_path, net: nn.Module):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = Config(args)

    cfg.trainset = args.trainset
    cfg.testset = args.testset
    cfg.multiple_snr = args.multiple_snr
    cfg.C = args.C
    cfg.model = args.model
    cfg.model_size = args.model_size

    print("Batch size:", cfg.batch_size)

    seed_torch(cfg.seed)
    logger = configure_logger(cfg)
    logger.info(f"Configuration: {cfg.__dict__}")

    train_loader, test_loader = get_loader(args, cfg)

    net = SwinJSCC(args, cfg).to(cfg.device)

    pretrained_path = "./checkpoint/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    if Path(pretrained_path).exists():
        net.load_state_dict(torch.load(pretrained_path, map_location=cfg.device))
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    trainer = Trainer(cfg, net, train_loader, test_loader, logger)

    if args.training:
        start_epoch = 0
        for epoch in range(start_epoch, cfg.total_epochs):
            trainer.train_one_epoch(epoch)
            if (epoch + 1) % cfg.save_model_freq == 0:
                trainer.save_checkpoint(epoch + 1)
                trainer.evaluate()
    else:
        trainer.evaluate()

if __name__ == "__main__":
    main()



