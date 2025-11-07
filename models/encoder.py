from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import loralib as lora

from models.pos_embed import interpolate_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

    '''
    modified from https://github.com/huggingface/pytorch-image models/blob/main/timm/models/vision_transformer.py#L826
    '''
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def _is_full_ft(ft_blks):
    return isinstance(ft_blks, str) and ft_blks.lower() == "full"

def build_model(args):
    '''
    args requires:
    - arch
    - img_size
    - ft
    - ft_blks  # int or string "full"
    - lora_rank
    '''
    path = Path('/home/hch/dementia/models/weights/')
    fn = f'{args.arch}.pth.tar'
    ckpt = torch.load(path/fn, map_location='cpu')

    if args.arch == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
    elif args.arch == 'dinov3':
        model = torch.hub.load('/home/hch/dementia/dinov3', 'dinov3_vitl16', source='local',
                               weights='/home/hch/dementia/models/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                               img_size=args.img_size)
    elif args.arch == 'openclip':
        model = timm.create_model('vit_large_patch14_clip_224.laion2b', pretrained=False, num_classes=0,
                                  img_size=args.img_size, dynamic_img_size=True)
    elif args.arch == 'mae':
        model = timm.create_model('vit_large_patch16_224.mae', pretrained=False, num_classes=0,
                                  img_size=args.img_size, dynamic_img_size=True)
    elif args.arch == 'retfound':
        model = RETFound_mae(img_size=args.img_size, num_classes=0, global_pool='token')
        interpolate_pos_embed(model, ckpt)
    elif args.arch == 'retfound_dinov2':
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False,
                                  img_size=args.img_size, num_classes=0)

    if args.ft != 'lora':
        model.load_state_dict(ckpt)

    freeze(model)

    if args.ft == 'lora':
        unfreeze(model.norm)

        blocks = list(model.blocks)
        if _is_full_ft(args.ft_blks):
            target_blocks = blocks
        else:
            n = int(args.ft_blks)
            target_blocks = list(reversed(blocks))[:n]

        for blk in target_blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
                in_f = blk.attn.qkv.in_features
                out_f = blk.attn.qkv.out_features
                blk.attn.qkv = lora.MergedLinear(
                    in_f, out_f,
                    r=args.lora_rank,
                    lora_alpha=args.lora_rank,
                    enable_lora=[True, False, True],
                    merge_weights=False
                )

        model.load_state_dict(ckpt, strict=False)

        lora.mark_only_lora_as_trainable(model)

    elif args.ft == 'partial':
        unfreeze(model.norm)
        for i, blk in enumerate(reversed(list(model.blocks))):
            if i >= args.ft_blks:
                break
            unfreeze(blk)

    model.head = nn.Linear(1024, 1)
    return model
