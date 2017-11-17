import torch
from torch.nn import functional


def sequence_mask(length, max_length=None):
    """
    Args:
        length (Tensor): A long tensor of size (batch_size,).
        max_length (int): The maximum length. If None, it automatically
            sets this as max(lengths).
    Returns:
        mask: (Tensor): A byte mask tensor of size
            (max_length, batch_size). Each element is 1 if valid
            and 0 else.
    """

    if max_length is None:
        max_length = length.max()
    seq_range = torch.arange(0, max_length).unsqueeze(1).long()
    if length.is_cuda:
        device = length.get_device()
        seq_range = seq_range.cuda(device)
    length = length.unsqueeze(0)
    mask = torch.lt(seq_range, length)
    return mask


def sequence_cross_entropy(logits, targets, length):
    log_probs = functional.log_softmax(logits, dim=2)
    losses = -torch.gather(log_probs, dim=2, index=targets.unsqueeze(2))
    losses = losses.squeeze(2)
    mask = sequence_mask(length=length, max_length=logits.size(0))
    losses.data.masked_fill_(mask=~mask, value=0)
    loss = losses.sum() / losses.size(1)
    return loss
