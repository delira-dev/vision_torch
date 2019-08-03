import torch


def make_onehot_torch(labels, n_classes):
    dtype = labels.dtype
    labels = labels.view(-1, 1).long()

    labels_onehot = torch.zeros(labels.size(0), n_classes,
                                device=labels.device, dtype=dtype)
    labels_onehot.scatter_(1, labels, 1)
    return labels_onehot
