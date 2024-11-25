"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Code from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [batch_size, num_views, ...].
            labels: ground truth of shape [batch_size].
            mask: contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self._get_device(features)
        features = self._check_and_adjust_features(features)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        mask = self._get_mask(features, labels, mask, device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature, anchor_count = self._get_anchor_features(contrast_feature, features)

        logits = self._compute_logits(anchor_feature, contrast_feature)
        mask = self._get_updated_mask(mask, anchor_count, contrast_count, device)

        log_prob = self._compute_log_prob(logits, mask, device)
        loss = self._compute_loss(log_prob, mask, anchor_count, features.shape[0])

        return loss

    def _get_device(self, features):
        """Determine the device (CPU or CUDA) based on the features tensor."""
        return torch.device('cuda') if features.is_cuda else torch.device('cpu')

    def _check_and_adjust_features(self, features):
        """Ensure features tensor has the correct shape, adjust if necessary."""
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, num_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        return features

    def _get_mask(self, features, labels, mask, device):
        """Generate or validate the contrastive mask."""
        batch_size = features.shape[0]
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        return mask

    def _get_anchor_features(self, contrast_feature, features):
        """Get anchor features based on contrast mode."""
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = features.shape[1]
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')
        return anchor_feature, anchor_count

    def _compute_logits(self, anchor_feature, contrast_feature):
        """Compute the logits for contrastive learning."""
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        return anchor_dot_contrast - logits_max.detach()

    def _get_updated_mask(self, mask, anchor_count, contrast_count, device):
        """Update the mask to exclude self-contrast cases and repeat it accordingly."""
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        return mask * logits_mask

    def _compute_log_prob(self, logits, mask, device):
        """Compute the log probability."""
        exp_logits = torch.exp(logits) * mask
        return logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  # Adding small value for numerical stability

    def _compute_loss(self, log_prob, mask, anchor_count, batch_size):
        """Compute the final contrastive loss."""
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)  # Adding small value to avoid division by zero
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
