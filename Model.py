from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

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