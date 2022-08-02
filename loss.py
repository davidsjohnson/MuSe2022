import torch
import torch.nn as nn

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y, seq_lens=None):
        return torch.sqrt(self.mse(yhat, y))

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None):
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)

        # reduce mean and var without original dimensions so calculation works with mbp-stress data #TODO Validate this doesn't affect baseline data
        y_true_mean = torch.sum(y_true * mask) / torch.sum(mask)
        y_pred_mean = torch.sum(y_pred * mask) / torch.sum(mask)

        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2) / torch.sum(mask)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2) / torch.sum(mask)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean)) / torch.sum(mask)

        ccc = 2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)

        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss

# wraps BCEWithLogitsLoss, but constructor accepts (and ignores) argument seq_lens
class WrappedBCEWithLogitsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


class WrappedMSELoss(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


# wraps BCELoss, but constructor accepts (and ignores) argument seq_lens
class WrappedBCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)



def get_segment_wise_labels(labels):
    # collapse labels to one label per segment (as originally for MuSe-Sent)
    segment_labels = []
    for i in range(labels.size(0)):
        segment_labels.append(labels[i, 0, :])
    labels = torch.stack(segment_labels).long()
    return labels


def get_segment_wise_logits(logits, feature_lens):
    # determines exactly one output for each segment (by taking the last timestamp of each segment)
    segment_logits = []
    for i in range(logits.size(0)):
        segment_logits.append(logits[i, feature_lens[i] - 1, :])  # (batch-size, frames, classes)
    logits = torch.stack(segment_logits, dim=0)
    return logits
