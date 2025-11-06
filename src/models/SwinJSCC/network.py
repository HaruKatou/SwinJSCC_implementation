from SwinJSCC.decoder import *
from SwinJSCC.encoder import *
from SwinJSCC.channel import Channel
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
    
    def feature_pass_channel(self, feature, snr, avg_power=False):
        """Pass features through channel (AWGN or Rayleigh)."""
        return self.channel.forward(feature, snr, avg_power)
    
    def _update_resolution(self, H, W):
        """Update encoder/decoder internal resolution when input size changes."""
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H, self.W = H, W

    def _select_channel_params(self, given_SNR=None, given_rate=None):
        """Randomly select or use provided SNR and channel rate."""
        SNR = given_SNR if given_SNR is not None else choice(self.multiple_snr)
        channel_num = given_rate if given_rate is not None else choice(self.channel_number)
        return SNR, channel_num
    
    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        self._update_resolution(H, W)
        snr, channel_num = self._select_channel_params(given_SNR, given_rate)

        # Encode
        if self.model in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA']:
            feature = self.encoder(input_image, snr, channel_num, self.model)
            CBR = feature.numel() / (2 * input_image.numel())

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, snr)
            else:
                noisy_feature = feature

        elif self.model in ['SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA']:
            feature, mask = self.encoder(input_image, snr, channel_num, self.model)
            CBR = channel_num / (2 * 3 * 2 ** (self.downsample * 2))
            avg_power = (feature ** 2).sum() / mask.sum()

            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, snr, avg_power)
            else:
                noisy_feature = feature

            noisy_feature = noisy_feature * mask

        else:
            raise ValueError(f"Unknown model variant: {self.model}")
        
        # Decode
        recon_image = self.decoder(noisy_feature, snr, self.model)

        # Compute metrics
        mse = self.mse_loss(input_image * 255., recon_image.clamp(0., 1.) * 255.).mean()
        loss = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.)).mean()

        return recon_image, CBR, snr, mse, loss
