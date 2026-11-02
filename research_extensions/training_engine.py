"""Shared ABMIL training and hard-target evaluation routines."""
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from dataloader import mil_collate_fn
from utils import classification_metrics

def make_loaders(train_data, val_data, test_data, conf):
    common = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker),
                  pin_memory=bool(conf.pin_memory) and torch.cuda.is_available(),
                  collate_fn=mil_collate_fn)
    # A separate generator keeps the training order reproducible across ablations.
    generator = torch.Generator().manual_seed(int(conf.seed))
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32, non_blocking=True)
        labels = data['label'].to(device, dtype=torch.long, non_blocking=True)
        mask = data['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError('Training loader is empty.')
    return {'loss': total_loss / total_examples, 'acc': 100.0 * total_correct / total_examples}

def calibration_metrics(probabilities, labels, n_bins=15):
    # Top-label ECE: fixed equal-width bins; Brier: sum over classes, mean over slides.
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    bin_index = (confidence * n_bins).long().clamp(max=n_bins - 1)
    ece = torch.zeros((), dtype=probabilities.dtype)
    for i in range(n_bins):
        selected = bin_index == i
        if selected.any():
            ece += selected.float().mean() * (correct[selected].mean() - confidence[selected].mean()).abs()
    targets = F.one_hot(labels, num_classes=probabilities.shape[1]).float()
    brier = (probabilities - targets).square().sum(dim=1).mean()
    return {'ece': float(ece), 'brier': float(brier)}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, return_predictions=False):
    model.eval()
    total_loss, total_examples = 0.0, 0
    probabilities_all, labels_all, slide_ids = [], [], []
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32, non_blocking=True)
        labels = data['label'].to(device, dtype=torch.long, non_blocking=True)
        mask = data['mask'].to(device, non_blocking=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        probabilities = F.softmax(logits, dim=1)
        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probabilities_all.append(probabilities.cpu())
        labels_all.append(labels.cpu())
        slide_ids.extend(data['slide_id'])
    if total_examples == 0:
        raise ValueError('Evaluation loader is empty.')
    probabilities_all = torch.cat(probabilities_all)
    labels_all = torch.cat(labels_all)
    metrics = classification_metrics(probabilities_all, labels_all, conf.n_class)
    metrics['loss'] = total_loss / total_examples
    metrics.update(calibration_metrics(probabilities_all, labels_all))
    if return_predictions:
        predictions = pd.DataFrame({'slide_id': slide_ids, 'label': labels_all.numpy(),
                                    'prediction': probabilities_all.argmax(1).numpy()})
        for c in range(int(conf.n_class)):
            predictions[f'p_{c}'] = probabilities_all[:, c].numpy()
        return metrics, predictions
    return metrics
