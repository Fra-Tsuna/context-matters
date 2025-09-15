from abc import ABC, abstractmethod


class Agent(ABC):
    
    def __init__(self, model):
        self.model = model

    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def llm_call(self, prompt, question, **kwargs) -> str:
        pass