from numpy import log, prod


class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = prod(x.size()[1:])
    bpd = nll / (log(2) * dim)

    return bpd
