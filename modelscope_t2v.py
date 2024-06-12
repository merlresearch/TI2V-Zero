# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
#
# Code adapted from https://github.com/modelscope/modelscope/tree/57791a8cc59ccf9eda8b94a9a9512d9e3029c00b/modelscope/models/multi_modal/video_synthesis -- Apache 2.0 License

from os import path as osp

import numpy as np
import open_clip
import torch
import torch.cuda.amp as amp
from einops import rearrange
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

# from modelscope.models.multi_modal.video_synthesis.autoencoder import \
#     AutoencoderKL
from autoencoder import AutoencoderKL
from diffusion import GaussianDiffusion, beta_schedule
from unet_sd import UNetSD

__all__ = ["TextToVideoSynthesis"]


@MODELS.register_module(Tasks.text_to_video_synthesis, module_name=Models.video_synthesis)
class TextToVideoSynthesis(Model):
    r"""
    task for text to video synthesis.

    Attributes:
        sd_model: denosing model using in this task.
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation into visual space with VQGAN.
        clip_encoder: encode the text into text embedding.
    """

    def __init__(self, model_dir, *args, **kwargs):
        r"""
        Args:
            model_dir (`str` or `os.PathLike`)
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co
                      or modelscope.cn. Valid model ids can be located at the root-level, like `bert-base-uncased`,
                      or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
        """
        super().__init__(model_dir=model_dir, *args, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = Config.from_file(osp.join(model_dir, ModelFile.CONFIGURATION))
        cfg = self.config.model.model_cfg
        cfg["temporal_attention"] = True if cfg["temporal_attention"] == "True" else False

        print("Num GPUs available: ", torch.cuda.device_count())

        # Initialize unet
        self.sd_model = UNetSD(
            in_dim=cfg["unet_in_dim"],
            dim=cfg["unet_dim"],
            y_dim=cfg["unet_y_dim"],
            context_dim=cfg["unet_context_dim"],
            out_dim=cfg["unet_out_dim"],
            dim_mult=cfg["unet_dim_mult"],
            num_heads=cfg["unet_num_heads"],
            head_dim=cfg["unet_head_dim"],
            num_res_blocks=cfg["unet_res_blocks"],
            attn_scales=cfg["unet_attn_scales"],
            dropout=cfg["unet_dropout"],
            temporal_attention=cfg["temporal_attention"],
        )
        if kwargs["retrained_path"] is not None:
            trained_ckpt = torch.load(kwargs["retrained_path"], map_location="cpu")
            sd_model_ckpt = self.sd_model.state_dict()
            for name, param in sd_model_ckpt.items():
                sd_model_ckpt[name].copy_(trained_ckpt["state_dict"]["model.diffusion_model." + name])
            self.sd_model.load_state_dict(sd_model_ckpt, strict=True)
        else:
            self.sd_model.load_state_dict(
                torch.load(osp.join(model_dir, self.config.model.model_args.ckpt_unet)), strict=True
            )
        self.sd_model.eval()
        self.sd_model.to(self.device)

        # Initialize diffusion
        betas = beta_schedule("linear_sd", cfg["num_timesteps"], init_beta=0.00085, last_beta=0.0120)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg["mean_type"],
            var_type=cfg["var_type"],
            loss_type=cfg["loss_type"],
            rescale_timesteps=False,
        )

        # Initialize autoencoder
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
        ckpt_autoenc_path = osp.join(model_dir, self.config.model.model_args.ckpt_autoencoder)
        if kwargs["finetune_autoenc"] is not None:
            ckpt_autoenc_path = kwargs["finetune_autoenc"]
        self.autoencoder = AutoencoderKL(ddconfig, 4, ckpt_autoenc_path)
        if self.config.model.model_args.tiny_gpu == 1:
            self.autoencoder.to("cpu")
        else:
            self.autoencoder.to(self.device)
        self.autoencoder.eval()

        # Initialize Open clip
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            version=osp.join(model_dir, self.config.model.model_args.ckpt_clip), layer="penultimate"
        )
        if self.config.model.model_args.tiny_gpu == 1:
            self.clip_encoder.to("cpu")
        else:
            self.clip_encoder.to(self.device)

    @torch.no_grad()
    def forward(self, input, ddim_step=50, guide_scale=9.0):
        r"""
        The entry function of text to image synthesis task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using vqgan model (autoencoder) to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """
        y = input["text_emb"]
        zero_y = input["text_emb_zero"]
        context = torch.cat([zero_y, y], dim=0).to(self.device)
        # synthesis
        with torch.no_grad():
            num_sample = 1  # here let b = 1
            max_frames = self.config.model.model_args.max_frames
            latent_h, latent_w = 32, 32
            with amp.autocast(enabled=True):
                x0 = self.diffusion.ddim_sample_loop(
                    noise=torch.randn(num_sample, 4, max_frames, latent_h, latent_w).to(
                        self.device
                    ),  # shape: b c f h w
                    model=self.sd_model,
                    model_kwargs=[
                        {"y": context[1].unsqueeze(0).repeat(num_sample, 1, 1)},
                        {"y": context[0].unsqueeze(0).repeat(num_sample, 1, 1)},
                    ],
                    guide_scale=guide_scale,
                    ddim_timesteps=ddim_step,
                    eta=0.0,
                )

                scale_factor = 0.18215
                video_data = 1.0 / scale_factor * x0
                bs_vd = video_data.shape[0]
                video_data = rearrange(video_data, "b c f h w -> (b f) c h w")
                self.autoencoder.to(self.device)
                video_data = self.autoencoder.decode(video_data)
                if self.config.model.model_args.tiny_gpu == 1:
                    self.autoencoder.to("cpu")
                video_data = rearrange(video_data, "(b f) c h w -> b c f h w", b=bs_vd)
        return video_data.type(torch.float32).cpu()

    @torch.no_grad()
    def forward_with_vid_resample(
        self, input, vid, add_vid_cond, use_ddpm_inversion, resample_iter, ddim_step=50, guide_scale=9.0
    ):
        y = input["text_emb"]
        zero_y = input["text_emb_zero"]
        bs = y.size(0)
        self.autoencoder.to(self.device)
        scale_factor = 0.18215
        vid_embedding = None
        if vid is not None:
            vid = vid.to(self.device)
            # encode video to embedding
            vid_embedding = []
            for frame_idx in range(vid.size(2)):
                img = vid[:, :, frame_idx]
                img_embedding = self.autoencoder.encode(img).mean
                img_embedding = scale_factor * img_embedding
                visual_rec_image = False
                if visual_rec_image:
                    rec_img = self.autoencoder.decode(1.0 / scale_factor * img_embedding)
                    mean = [0.5, 0.5, 0.5]
                    std = [0.5, 0.5, 0.5]
                    mean = torch.tensor(mean, device=rec_img.device).reshape(1, -1, 1, 1)  # nchw
                    std = torch.tensor(std, device=rec_img.device).reshape(1, -1, 1, 1)  # nchw
                    rec_img = rec_img.mul_(std).add_(mean)
                    rec_img.clamp_(0, 1)
                    rec_img_data = np.array(rec_img[0].permute(1, 2, 0).data.cpu().numpy() * 255.0, dtype=np.uint8)
                vid_embedding.append(img_embedding)
            vid_embedding = torch.stack(vid_embedding, dim=2)

        # synthesis
        with torch.no_grad():
            num_sample = bs  # here let b = 1
            max_frames = self.config.model.model_args.max_frames
            latent_h, latent_w = 32, 32
            with amp.autocast(enabled=True):
                x0 = self.diffusion.ddim_sample_loop_with_vid_resample(
                    noise=torch.randn(num_sample, 4, max_frames, latent_h, latent_w).to(self.device),
                    # shape: b c f h w
                    model=self.sd_model,
                    model_kwargs=[
                        {"y": y.to(self.device)},
                        {"y": zero_y.to(self.device)},
                    ],
                    cond_vid=vid_embedding,
                    guide_scale=guide_scale,
                    ddim_timesteps=ddim_step,
                    eta=0.0,
                    add_vid_cond=add_vid_cond,
                    use_ddpm_inversion=use_ddpm_inversion,
                    resample_iter=resample_iter,
                )

                video_data = 1.0 / scale_factor * x0
        return video_data.type(torch.float32).cpu()


class FrozenOpenCLIPEmbedder(torch.nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="open_clip_pytorch_model.bin",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device("cpu"), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
