import torch


def mmap_collate_fn(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([])

    features, targets = zip(*batch)
    batch_features = torch.stack(features)

    if isinstance(targets[0], tuple):
        win_probs, original_results = zip(*targets)
        batch_win_probs = torch.stack(win_probs)
        batch_original_results = torch.stack(original_results)
        return batch_features, (batch_win_probs, batch_original_results)
    else:
        batch_targets = torch.stack(targets)
        return batch_features, batch_targets
