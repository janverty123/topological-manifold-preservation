"""
models.py
---------
Defines the neural network used across all three training regimes
(Finetune, EWC, TMP). The Mathematical Formulation section refers to
"a specific hidden layer [that] act[s] as a mapping function that
projects an input sample into a high-dimensional feature space" --
this is `hidden2` below (128-d), which is registered with a forward
hook so its activations can be extracted for topological analysis.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden1_dim=256, hidden2_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.act2 = nn.ReLU()  # output of this layer is the monitored feature space
        self.fc3 = nn.Linear(hidden2_dim, num_classes)

        # populated by the forward hook each forward pass
        self._last_hidden2_activation = None

        self.act2.register_forward_hook(self._capture_hidden2)

    def _capture_hidden2(self, module, inp, output):
        self._last_hidden2_activation = output

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        logits = self.fc3(h2)
        return logits

    def get_last_hidden_activation(self):
        """Returns the most recent hidden2 activation tensor (post forward pass)."""
        if self._last_hidden2_activation is None:
            raise RuntimeError("No forward pass has been run yet.")
        return self._last_hidden2_activation

    @torch.no_grad()
    def extract_activations(self, dataloader, device, max_samples=None):
        """
        Runs a full pass over `dataloader` and collects hidden2 activations
        (detached, on CPU) for downstream topological data analysis.
        Used to build the activation point cloud referenced in the
        Mathematical Formulation ("resulting collection of activation
        vectors forms a discrete geometric point cloud").
        """
        self.eval()
        activations = []
        collected = 0
        for x, _ in dataloader:
            x = x.to(device)
            _ = self.forward(x)
            h2 = self.get_last_hidden_activation().detach().cpu()
            activations.append(h2)
            collected += h2.shape[0]
            if max_samples is not None and collected >= max_samples:
                break
        return torch.cat(activations, dim=0)[:max_samples] if max_samples else torch.cat(activations, dim=0)


def apply_class_mask(logits, allowed_classes, device):
    """
    Masks out logits for classes that are not part of the current task by
    setting them to -inf, so cross-entropy / softmax only competes among
    the allowed classes. This implements class-incremental Split-MNIST
    training with a single shared output head.
    """
    mask = torch.full((logits.shape[1],), float("-inf"), device=device)
    mask[allowed_classes] = 0.0
    return logits + mask.unsqueeze(0)
