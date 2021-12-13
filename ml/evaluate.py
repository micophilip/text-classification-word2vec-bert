import torch
from metrics import binary_accuracy, get_metrics, get_rounded_preds, get_confidences_tensor, get_sigmoid


def evaluate(model, iterator, criterion, test_mode=False):
    epoch_loss = 0
    epoch_acc = 0
    y_all = []
    y_hat_all = []
    confidences = []
    sample_ids = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch['text']
            text_lengths = batch['text_length']

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch['label'])

            if test_mode:
                y_all.extend(batch['label'].tolist())
                y_hat_all.extend(get_rounded_preds(predictions).tolist())
                confidences.extend(get_confidences_tensor(get_sigmoid(predictions)).tolist())
                sample_ids.extend(batch['sample_id'])

            acc = binary_accuracy(predictions, batch['label'])

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    metrics = get_metrics(y_hat_all, y_all, sample_ids, confidences) if test_mode else {}

    return epoch_loss / len(iterator), epoch_acc / len(iterator), metrics
