from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
import os

from .base_strategy import BaseStrategy

class LoRAStrategy(BaseStrategy):
    def __init__(self, r=32, alpha=16, dropout=0.05, target_modules=None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ['q_proj', 'v_proj']

    def setup_model(self, model, tokenizer):
        # Prepare model for k-bit training if needed
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            target_modules=self.target_modules
        )
        model = get_peft_model(model, lora_config)
        return model

    def train(self, model, tokenizer, train_dataset, eval_dataset, output_dir, **kwargs):
        # This is a simplified Trainer usage example. Adjust for large models / accelerate.
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=kwargs.get('batch_size', 1),
            per_device_eval_batch_size=1,
            num_train_epochs=kwargs.get('epochs', 1),
            fp16=kwargs.get('fp16', True),
            save_strategy='epoch',
            logging_strategy='steps',
            logging_steps=50,
            remove_unused_columns=False
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        trainer.train()
        model.save_pretrained(output_dir)
        return output_dir
