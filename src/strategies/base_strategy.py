from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def setup_model(self, model, tokenizer):
        pass

    def train(self, model, tokenizer, train_dataset, eval_dataset, output_dir, **kwargs):
        raise NotImplementedError
