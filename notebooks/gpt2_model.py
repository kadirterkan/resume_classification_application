from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from base_classes import compute_metrics


class GPT2:
    def __init__(self, num_labels):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn't have a pad token, so we use the eos token

        self.model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
        self.model.config.pad_token_id = self.model.config.eos_token_id  # Set pad token ID to eos token ID

        self.training_args = TrainingArguments(
            output_dir="./result",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            num_train_epochs=3,
            report_to="none",
        )

    def set_trainer(self, train_dataset, val_dataset):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )