import os
from tqdm import tqdm
import PIL
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
from ..utils import read_image_rgb


logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, objects, classifier_transforms):
        self.objects = objects
        self.classifier_transforms = classifier_transforms

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]
        assert isinstance(obj, (str, np.ndarray, Image.Image))

        if isinstance(obj, str):
            pil_img = read_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            pil_img = Image.fromarray(obj)
        elif isinstance(obj, Image.Image):
            pil_img = obj

        resnet_img = self.classifier_transforms(pil_img).float()

        return resnet_img


class WatermarksPredictor:
    def __init__(self, wm_model, classifier_transforms, device):
        if not os.path.exists(wm_model):
            raise ValueError(f"Must provide a valid path to the ONNX model file got {wm_model}.")

        if wm_model.endswith(".onnx"):
            self.use_onnx = True
        else:
            self.use_onnx = False

        self.wm_model = wm_model
        if not self.use_onnx:
            self.wm_model.eval()
        self.classifier_transforms = classifier_transforms
        self.device = device
        if self.use_onnx:
            import onnxruntime

            logger.info("Setting device to CPU because `use_onnx` is True.")
            self.device = "cpu"

            self.session = onnxruntime.InferenceSession(wm_model)
            logger.info("ONNX session initialized.")
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.map = {0: "clean", 1: "watermarked"}

    def predict_image(self, pil_image):
        pil_image = pil_image.convert("RGB")
        input_img = self.classifier_transforms(pil_image).float().unsqueeze(0)
        outputs = self.wm_model(input_img.to(self.device))
        result = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()[0]
        return result

    def predict_image_with_onnx(self, pil_image, return_probs=False):
        if isinstance(pil_image, PIL.Image.Image):
            input_data = self.classifier_transforms(pil_image).float().unsqueeze(0).numpy()
        else:
            input_data = pil_image.numpy()
        result = self.session.run([self.output_name], {self.input_name: input_data})[0]
        if return_probs:
            return torch.nn.functional.softmax(torch.from_numpy(result), dim=-1).numpy()
        else:
            predicted_classes = np.argmax(result, axis=-1).tolist()
            return predicted_classes

    def run(self, files, num_workers=1, bs=8, pbar=True, return_probs=False):
        if return_probs and not self.use_onnx:
            raise ValueError("`return_probs` can only be True when using an ONNX model.")

        eval_dataset = ImageDataset(files, self.classifier_transforms)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=bs,
            drop_last=False,
            num_workers=num_workers,
        )
        if pbar:
            loader = tqdm(loader)

        result = []
        for batch in loader:
            if not self.use_onnx:
                with torch.no_grad():
                    outputs = self.wm_model(batch.to(self.device))
                    result.extend(torch.max(outputs, 1)[1].cpu().reshape(-1).tolist())
            else:
                output = self.predict_image_with_onnx(batch, return_probs)
                if return_probs:
                    result.extend(output)
                else:
                    result.extend([self.map[pred_class] for pred_class in output])

        # Only return the softmax scores associated to watermark
        return np.stack(result)[:, 1] if return_probs else result
