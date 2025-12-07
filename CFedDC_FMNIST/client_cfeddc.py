# client_cfeddc.py
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class CFedDCClient:
    """
    CFedDC-style client training:
      L = CE + tau * [ lam_t * ||w - w_head||^2/2 + (1-lam_t) * ||w - w_prev||^2/2 ]
    RF: starts from head weights
    RL: can start from exchanged RF weights (if available), but regularizes to the chosen head
    """
    def __init__(self, cid, model_fn, train_subset, val_subset, device, batch_size=64, lr=1e-3, local_epochs=1):
        self.cid = cid
        self.device = device
        self.model = model_fn().to(device)
        self.old_model = copy.deepcopy(self.model)
        self.exchange_model = copy.deepcopy(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.local_epochs = local_epochs
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # ---- states & vectors ----
    @torch.no_grad()
    def acc_on_val(self):
        self.model.eval()
        correct, total = 0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = torch.argmax(self.model(x), dim=1)
            correct += (preds == y).sum().item() 
            total += y.size(0)
        return correct / max(1, total)
    

    def get_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    def set_state(self, state_dict): self.model.load_state_dict(state_dict)
    def load_exchange_from(self, other_state): self.exchange_model.load_state_dict(other_state)

    @torch.no_grad()
    def get_param_vec(self):
        return torch.cat([p.detach().flatten().cpu() for p in self.model.parameters()])

    # ---- metrics ----
    @torch.no_grad()
    def f1_on_val(self):
        self.model.eval()
        ys, ps = [], []
        for x, y in self.val_loader:
            x = x.to(self.device)
            p = torch.argmax(self.model(x), dim=1).cpu()
            ps.append(p); ys.append(y)
        ys = torch.cat(ys).numpy(); ps = torch.cat(ps).numpy()
        return float(f1_score(ys, ps, average="macro"))

    @torch.no_grad()
    def val_loss_with_head(self, head_state):
        ce = nn.CrossEntropyLoss()
        temp = copy.deepcopy(self.model).to(self.device)
        temp.load_state_dict(head_state); temp.eval()
        tot, n = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = temp(x)
            tot += ce(out, y).item() * x.size(0)
            n += x.size(0)
        return tot / max(1, n)

    # ---- losses ----
    def _vec(self, params): return torch.cat([p.view(-1) for p in params])
    def _sq_l2(self, params_a, params_b): return 0.5 * torch.sum((self._vec(params_a) - self._vec(params_b))**2)

    def _train_core(self, head_state, lam_t, tau):
        ce = nn.CrossEntropyLoss()
        head_model = copy.deepcopy(self.model).to(self.device)
        head_model.load_state_dict(head_state)

        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                out = self.model(x)
                loss_ce = ce(out, y)
                sim = self._sq_l2(self.model.parameters(), head_model.parameters())
                stab = self._sq_l2(self.model.parameters(), self.old_model.parameters())
                loss = loss_ce + tau * (lam_t * sim + (1 - lam_t) * stab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

        # refresh stability anchor
        self.old_model.load_state_dict(self.model.state_dict())

    def train_RF(self, head_state, lam_min, lam_max, t, T, tau=1.0):
        # start from head
        self.set_state(head_state)
        lam_t = lam_min + (lam_max - lam_min) * (t / max(1, T-1))
        self._train_core(head_state, lam_t, tau)
        return self.get_state(), len(self.train_loader.dataset)

    def train_RL(self, head_state, lam_min, lam_max, t, T, tau=1.0):
        lam_t = lam_min + (lam_max - lam_min) * (t / max(1, T-1))
        self._train_core(head_state, lam_t, tau)
        return self.get_state(), len(self.train_loader.dataset)
