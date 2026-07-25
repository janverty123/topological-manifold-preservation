"""
ewc.py
------
Elastic Weight Consolidation (Kirkpatrick et al., 2017), used as one of
the two baselines the TMP algorithm is benchmarked against (Objective 4
/ Research Question 2.1-2.3: "Analyze the difference ... with Elastic
Weight Consolidation (EWC) and Finetune").

EWC penalizes changes to parameters that were important for Task 1,
where importance is estimated via the diagonal of the Fisher
Information Matrix computed on Task-1 data.
"""

import torch
import torch.nn.functional as F


class EWC:
    def __init__(self, model, task1_loader, device, allowed_classes, sample_size=500):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(task1_loader, allowed_classes, sample_size)

    def _compute_fisher(self, loader, allowed_classes, sample_size):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        mask = torch.full((10,), float("-inf"), device=self.device)
        mask[allowed_classes] = 0.0

        seen = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x) + mask.unsqueeze(0)
            log_probs = F.log_softmax(logits, dim=1)
            # sample labels from the model's predictive distribution
            # (standard Fisher-estimation procedure for EWC)
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            loss = F.nll_loss(log_probs, sampled)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * x.shape[0]

            seen += x.shape[0]
            if seen >= sample_size:
                break

        for n in fisher:
            fisher[n] /= max(seen, 1)
        return fisher

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss
