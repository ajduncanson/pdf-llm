from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    def query(self, prompt: str, context: str, model: str = None) -> str:
        pass
