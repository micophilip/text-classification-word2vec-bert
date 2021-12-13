import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def get_confidence(value: float) -> float:
    return value if value >= 0.5 else 1 - value


def get_sigmoid(preds):
    return torch.sigmoid(preds)


def get_confidences_tensor(sigmoids):
    return sigmoids.apply_(get_confidence)


def get_rounded_preds(preds):
    return torch.round(get_sigmoid(preds))


def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = get_rounded_preds(preds)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def get_metrics(y_hat, y, sample_ids, confidences):
    return {'precision': precision_score(y, y_hat), 'recall': recall_score(y, y_hat), 'f1': f1_score(y, y_hat),
            'classification_report': classification_report(y, y_hat), 'sample_id': sample_ids, 'confidence': confidences, 'y': y, 'y_hat': y_hat}
