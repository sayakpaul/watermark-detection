from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor

if __name__ == "__main__":
    transforms = get_watermarks_detection_model("convnext-tiny", fp16=False, device="cpu", return_transforms_only=True)
    predictor = WatermarksPredictor("convnext.onnx", transforms, use_onnx=True, device="cpu")

    result = predictor.predict_image_with_onnx(Image.open("images/watermark/1.jpg"))[0]
    print(predictor.map[result])

    results = predictor.run(
        [
            "images/watermark/1.jpg",
            "images/watermark/2.jpg",
            "images/watermark/3.jpg",
            "images/watermark/4.jpg",
            "images/clean/1.jpg",
            "images/clean/2.jpg",
            "images/clean/3.jpg",
            "images/clean/4.jpg",
        ],
        bs=4,
    )
    print(results)
