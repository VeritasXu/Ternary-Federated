import torch
import torch.nn.functional as F

def accuracy(true, pred, top_k=(1,)):

    max_k = max(top_k)
    batch_size = true.size(0)

    _, pred = pred.topk(max_k, 1)
    pred = pred.t()
    correct = pred.eq(true.view(1, -1).expand_as(pred))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.div_(batch_size).item())

    return result


def evaluate(model, loss, val_iterator, args):
    model.eval()
    model = model.to(args.device)
    loss_value = 0.0
    acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_iterator:
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            n_batch_samples = y_batch.size()[0]
            logits = model(x_batch)

            # compute logloss
            batch_loss = loss(logits, y_batch).item()

            # compute accuracies
            pred = F.softmax(logits, dim=1)
            batch_acc, batch_top5_acc = accuracy(y_batch, pred, top_k=(1, 5))

            loss_value += batch_loss * n_batch_samples
            acc += batch_acc * n_batch_samples
            top5_acc += batch_top5_acc * n_batch_samples
            total_samples += n_batch_samples

    return loss_value/total_samples, acc/total_samples, top5_acc/total_samples