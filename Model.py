from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        self.count = 0

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    @abstractmethod
    def get_GFLOPS(self):
        pass

    def get_total_FPS(self):
        return self.count/(self.preprocess_time+self.inference_time+self.postprocess_time)

    def get_inference_FPS(self):
        return self.count/self.inference_time
