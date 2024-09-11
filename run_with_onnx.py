import torch
from PIL import Image
import numpy as np
import onnxruntime
from wmdetection.models import get_watermarks_detection_model

MAP = {0: "clean", 1: "watermarked"}


# Step 1: Define and export the PyTorch model to ONNX
def export_model(model):
    model_path = "convnext.onnx"
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    # Define dynamic axes
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},  # Variable batch size, height and width
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


# Step 2: Prepare image for inference
def prepare_image(image_path, transforms):
    image = Image.open(image_path)
    return transforms(image).unsqueeze(0).numpy()


# Step 3: Run inference with ONNX Runtime
def run_inference(image_path, model_path, transforms):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data = np.stack([prepare_image(image_path, transforms) for _ in range(5)]).squeeze(1)
    result = session.run([output_name], {input_name: input_data})
    return result[0]


# Main execution
if __name__ == "__main__":
    # Export the model (only need to do this once)
    model, transforms = get_watermarks_detection_model("convnext-tiny", device="cpu", fp16=False)
    model_path = export_model(model)

    # Run inference
    # image_path = "dataset/synthetic_wm/_images_clean_1.jpg"
    image_path = "images/clean/2.jpg"
    output = run_inference(image_path, model_path, transforms)

    # Get the predicted class
    predicted_classes = np.argmax(output, axis=-1).tolist()
    for predicted_class in predicted_classes:
        print(f"Predicted class: {MAP[predicted_class]}")
