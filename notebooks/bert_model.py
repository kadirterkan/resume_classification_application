from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
from base_classes import compute_metrics


class Bert:
    def __init__(self, num_labels):
        self.train_dataset = None
        self.val_dataset = None
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                         num_labels=num_labels)

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