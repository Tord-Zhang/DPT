dependencies = ["torch"]

import numpy as np
import torch
import torch.nn as nn
import cv2
import util.io
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

def DPT(pretrained=True, model_type="dpt_hybrid", optimize=False):
    
    default_models = {
        "dpt_large": "dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "dpt_hybrid-midas-501f0c75.pt",
    }

    if model_type not in default_models.keys():
        raise ValueError("Only support model type dpt_large or dpt_hybrid, dpt_hybrid for default setting")
    
    state_dict = None
    if pretrained:
        checkpoint = (
            "https://github.com/Tord-Zhang/DPT/releases/download/torchhub/{}".format(
                default_models[model_type]
            )
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )

    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            state_dict=state_dict,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            state_dict=state_dict,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    
    model.eval()
    return model


def transforms(model_type="dpt_hybrid"):
    import cv2
    from torchvision.transforms import Compose
    from dpt.models import DPTDepthModel
    from dpt.midas_net import MidasNet_large
    from dpt.transforms import Resize, NormalizeImage, PrepareForNet


    if model_type == "dpt_large":  # DPT-Large
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError("Only support model type dpt_large or dpt_hybrid, dpt_hybrid for default setting")

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform


def read_image():
    return util.io.read_image