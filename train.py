from metrics import binary_accuracy


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text = batch['text']
        text_lengths = batch['text_length']
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch['label'])
        acc = binary_accuracy(predictions, batch['label'])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
