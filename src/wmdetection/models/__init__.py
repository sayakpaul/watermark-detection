import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from .convnext import ConvNeXt
from ..utils import FP16Module


def export_model(model):
    model_path = "convnext.onnx"
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    # Define dynamic axes
    dynamic_axes = {
        "input": {
            0: "batch_size",
            2: "height",
            3: "width",
        },  # Variable batch size, height and width
        "output": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        dynamic_axes=dynamic_axes,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
    )
    return model_path


def get_convnext_model(name):
    if name == "convnext-tiny":
        model_ft = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        model_ft.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=2),
        )

    detector_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model_ft, detector_transforms


def get_resnext_model(name):
    if name == "resnext50_32x4d-small":
        model_ft = models.resnext50_32x4d(pretrained=False)
    elif name == "resnext101_32x8d-large":
        model_ft = models.resnext101_32x8d(pretrained=False)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    detector_transforms = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return model_ft, detector_transforms


def get_watermarks_detection_model(name, device="cuda:0", fp16=True, pretrained=True, return_transforms_only=False):
    assert name in MODELS, f"Unknown model name: {name}"
    assert not (fp16 and name.startswith("convnext")), "Can`t use fp16 mode with convnext models"
    config = MODELS[name]

    model_ft, detector_transforms = config["constructor"](name)

    if pretrained and not return_transforms_only:
        path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        weights = torch.load(path, device, weights_only=True)
        model_ft.load_state_dict(weights)

    if fp16 and not return_transforms_only:
        model_ft = FP16Module(model_ft)

    if not return_transforms_only:
        model_ft.eval()
        model_ft = model_ft.to(device)

    return (model_ft, detector_transforms) if not return_transforms_only else detector_transforms


MODELS = {
    "convnext-tiny": dict(
        constructor=get_convnext_model,
        repo_id="boomb0om/watermark-detectors",
        filename="convnext-tiny_watermarks_detector.pth",
    ),
    "resnext101_32x8d-large": dict(
        constructor=get_resnext_model,
        repo_id="boomb0om/watermark-detectors",
        filename="watermark_classifier-resnext101_32x8d-input_size320-4epochs_c097_w082.pth",
    ),
    "resnext50_32x4d-small": dict(
        constructor=get_resnext_model,
        repo_id="boomb0om/watermark-detectors",
        filename="watermark_classifier-resnext50_32x4d-input_size320-4epochs_c082_w078.pth",
    ),
}
