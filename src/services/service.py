from src.services.model import ModelLoader
from src.services.inference import InferenceEngine


class Service:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.inference_engine = InferenceEngine()
