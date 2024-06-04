import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def tokenize_train_val(tokenizer, train_texts, val_texts, train_labels, val_labels):
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, train_labels.tolist())
    val_dataset = TextDataset(val_encodings, val_labels.tolist())

    return train_dataset, val_dataset
