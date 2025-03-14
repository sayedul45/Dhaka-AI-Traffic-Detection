import torch

class YOLOv5Inference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        # Load the YOLOv5 model using torch.hub with custom weights
        return torch.hub.load('./yolov5', 'custom', path=self.model_path, source='local')

    def run_inference(self, image_path):
        # Run inference on the provided image path
        results = self.model(image_path)
        results.print()  # Print results to console
        return results
