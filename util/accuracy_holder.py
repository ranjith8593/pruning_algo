"""Helper class for calculating and storing accuracy.
"""


class AccuracyHolder(object):
    """Computes and stores the average and current value"""
    avg = None
    val = 0
    sum = 0
    count = 0

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
