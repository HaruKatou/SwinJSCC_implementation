from .decoder import *
from .encoder import *
from .channel import Channel
from training.loss import Distortion
from random import choice
import torch.nn as nn

class SwinJSCC(nn.Module):
    """
    SwinJSCC: Joint Source-Channel Coding framework using Swin Transformer.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        channel (Channel): Channel simulator (AWGN, Rayleigh, etc.).
        distortion_loss (Distortion): Distortion metric.
    """
    
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        self.model = args.model
        self.pass_channel = config.pass_channel
        self.downsample = config.downsample

        self.encoder = create_encoder(**config.encoder_kwargs)
        self.decoder = create_decoder(**config.decoder_kwargs)

        self.channel = Channel(args, config)
        self.distortion_loss = Distortion(args)
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Parse multiple SNRs and channel numbers
        self.multiple_snr = [int(s) for s in args.multiple_snr.split(",")]
        self.channel_number = [int(c) for c in args.C.split(",")]

        self.H = self.W = 0

        # Logging
        if config.logger:
            logger = config.logger
            logger.info("=== SwinJSCC Configuration ===")
            logger.info(f"Encoder: {config.encoder_kwargs}")
            logger.info(f"Decoder: {config.decoder_kwargs}")

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss
    
    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature
    
    def _update_resolution(self, H, W):
        """Update encoder/decoder internal resolution when input size changes."""
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H, self.W = H, W
    
    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        self._update_resolution(H, W)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        # Encode
        if self.model in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA']:
            feature, _ = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = feature.numel() / (2 * input_image.numel())

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model in ['SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA']:
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
                
            noisy_feature = noisy_feature * mask

        else:
            raise ValueError(f"Unknown model variant: {self.model}")
        
        # Decode
        recon_image = self.decoder(noisy_feature, chan_param, self.model)

        # Compute metrics
        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()

        return recon_image, CBR, chan_param, mse, loss
