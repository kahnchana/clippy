import abc
import math

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from timm.models.vision_transformer import (
    VisionTransformer,
    build_model_with_cfg,
    checkpoint_filter_fn,
    checkpoint_seq,
    resolve_pretrained_cfg,
    PatchEmbed
)
from torch import Tensor, nn


class BlankLayer(nn.Module):
    pass


class CustomPatchEmbed(PatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CustomViT(VisionTransformer):
    def __init__(
            self,
            *args,
            image_pooling="gmp",
            embed_layer=CustomPatchEmbed,
            **kwargs,
    ):
        super(CustomViT, self).__init__(
            *args, embed_layer=embed_layer, **kwargs
        )
        self.image_pooling = image_pooling

    def forward_head(self, x, pre_logits: bool = False):
        if self.image_pooling:
            if self.image_pooling == "gap":
                x = x[:, self.num_prefix_tokens:].mean(dim=1)
            elif self.image_pooling == "gmp":
                x = x[:, self.num_prefix_tokens:].max(dim=-2)[0]
            elif self.image_pooling == "all":
                x = x[:, self.num_prefix_tokens:]
            else:  # cls by default
                x = x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, get_pos_tokens=False):
        x = self.forward_features(x, get_pos_tokens=get_pos_tokens)
        if get_pos_tokens:
            return self.fc_norm(x[:, self.num_prefix_tokens:])
        x = self.forward_head(x)
        return x

    def forward_features(self, x, get_pos_tokens=False):
        _, nc, h, w = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x, w, h)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x, w, h):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self._interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def _interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        CustomViT,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs,
    )
    return model


def vit_base_patch16_224(pretrained=False, variant="vit_base_patch16_224_dino", **kwargs):
    """ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(variant, pretrained=pretrained, **model_kwargs)
    return model


class CLIPpyModel(abc.ABC, torch.nn.Module):
    """ Implements code for running inference with pre-trained CLIPpy model.

    NOTE: weights used are for a model trained with lower batch-size leading to results below those in paper.
    """

    def __init__(
            self,
            image_pooling: str = "cls",
            text_pooling: str = "gap",
    ):
        super().__init__()

        self.visual = BlankLayer()

        self.visual.trunk = vit_base_patch16_224(True, image_pooling=image_pooling)

        self.text = SentenceTransformer("sentence-transformers/sentence-t5-base")
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.set_text_pooling(text_pooling)

        self._divisor_eps = 1e-4
        self._image_pooling = image_pooling
        self._text_pooling = text_pooling

    def forward(
            self,
            images: Tensor,
            input_ids: Tensor,
            input_id_masks: Tensor,
            get_pos_tokens: bool = False,
            **kwargs,
    ):

        image_encodings = self.encode_image(images, get_pos_tokens=get_pos_tokens)

        if get_pos_tokens:
            return {
                image_encodings: image_encodings,
            }

        text_encodings = self.encode_text(input_ids, input_id_masks)

        return {
            image_encodings: image_encodings,
            text_encodings: text_encodings,
        }

    def encode_text(self, input_ids: Tensor, input_id_masks: Tensor = None, **kwargs):
        output = self.text({"input_ids": input_ids, "attention_mask": input_id_masks})[
            "sentence_embedding"
        ]
        return self.text_head(output)

    def text_head(self, hidden_states: Tensor, input_id_masks: Tensor = None, **kwargs):
        return F.normalize(hidden_states, dim=-1, eps=self._divisor_eps).float()

    def encode_image(self, images: Tensor, get_pos_tokens: bool = False, **kwargs):
        output = self.visual.trunk(images, get_pos_tokens)
        return self.image_head(output, get_pos_tokens=get_pos_tokens)

    def image_head(self, hidden_states: Tensor, get_pos_tokens: bool = False, **kwargs):
        return F.normalize(hidden_states, dim=-1, eps=self._divisor_eps).float()

    def set_text_pooling(self, pooling):
        """ Converts pooling in the Hugging Face model to be max or average pooling"""
        if pooling == "gmp":
            self.text[1].pooling_mode_mean_tokens = False
            self.text[1].pooling_mode_max_tokens = True
        elif pooling == "gap":
            pass
        else:
            raise NotImplementedError(f"{pooling} not implemented")
