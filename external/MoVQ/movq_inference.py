import sys 
sys.path.append('glam_baseline/external/MoVQ')
from torch import nn
from .vqvae_blocks import Encoder, Decoder
from .quantize import VectorQuantizer2 as VectorQuantizer
from .movq_modules import MOVQDecoder
import os
import torch
from huggingface_hub import hf_hub_url, cached_download
from .configs import *
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import numpy as np
import requests

# import habana_frameworks.torch.core as htcore

class MOVQ(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 remap=None,
                 sane_index_shape=False):  # tell vector quantizer to return indices as bhw
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


def get_movqgan_model(name, pretrained=True, device='cuda', cache_dir='/tmp/movqgan', **model_kwargs):
    assert name in MODELS

    config = MODELS[name].copy()
    config['model_params'].update(model_kwargs)
    model = MOVQ(**config['model_params'])
    if pretrained:
        cache_dir = os.path.join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
        checkpoint = torch.load(os.path.join(cache_dir, config['filename']), map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = model.to(device)
    return model

def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = T.Compose([
            T.RandomResizedCrop(512, scale=(1., 1.), ratio=(1., 1.), interpolation=T.InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))

if __name__ == '__main__':
    model = get_movqgan_model('67M', pretrained=True, device=torch.device('hpu'))

    # url = "https://github.com/lxa9867/R2VOS/blob/master/illustration.jpg?raw=true"
    # response = requests.get(url)
    # img = prepare_image(Image.open(BytesIO(response.content)))

    # with torch.no_grad():
    #     out = model(img.to('hpu').unsqueeze(0))[0]
    #     # htcore.mark_step()
    #     code = model.encoder(img.to('hpu').unsqueeze(0))[0]
    #     # htcore.mark_step()

    # # verify
    # from torch.nn.functional import mse_loss, l1_loss
    # mse = mse_loss(out, img.to('hpu').unsqueeze(0))
    # l1 = l1_loss(out, img.to('hpu').unsqueeze(0))
    # print('mse =', np.round(mse.item(), 4), 'l1 =', np.round(l1.item(), 4))