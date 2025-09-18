# --- MovieLens model ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error
import numpy as np
import copy

class RatingsDataset(Dataset):
    """
    Wraps a ratings dataframe with columns: user_id, item_id, rating.
    If user_encoder/item_encoder are None, assumes user_id/item_id are already integer-encoded (0..n-1).
    """
    def __init__(self, df, user_encoder=None, item_encoder=None):
        u_col = df["user_id"].values
        i_col = df["item_id"].values
        r_col = df["rating"].values

        if user_encoder is not None:
            u_idx = user_encoder.transform(u_col)
        else:
            # assume already ints; cast safely
            u_idx = u_col.astype(np.int64, copy=False)

        if item_encoder is not None:
            i_idx = item_encoder.transform(i_col)
        else:
            i_idx = i_col.astype(np.int64, copy=False)

        self.u = torch.as_tensor(u_idx, dtype=torch.long)
        self.i = torch.as_tensor(i_idx, dtype=torch.long)
        self.r = torch.as_tensor(r_col, dtype=torch.float32)

    def __len__(self):
        return self.r.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        dot = (ue * ie).sum(dim=1)
        b = self.user_bias(u).squeeze(1) + self.item_bias(i).squeeze(1)
        return dot + b  # predicted rating

# --- CFedDC client (same structure; swap dataloaders, loss, metrics) ---

class CFedDCClient:
    def __init__(self, cid, model_fn, train_ds, val_ds, device, batch_size=1024, lr=1e-3, local_epochs=1):
        self.cid = cid
        self.device = device
        self.model = model_fn().to(device)
        self.old_model = copy.deepcopy(self.model)
        self.exchange_model = copy.deepcopy(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.local_epochs = local_epochs
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=True)

    def get_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    def set_state(self, state_dict): self.model.load_state_dict(state_dict)
    def load_exchange_from(self, state): self.exchange_model.load_state_dict(state)

    @torch.no_grad()
    def get_param_vec(self):
        return torch.cat([p.detach().flatten().cpu() for p in self.model.parameters()])

    @torch.no_grad()
    def rmse_on_val(self):
        self.model.eval()
        preds, trues = [], []
        for u, i, r in self.val_loader:
            u, i = u.to(self.device), i.to(self.device)
            y = self.model(u, i).cpu().numpy()
            preds.append(y); trues.append(r.numpy())
        y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
        return float(np.sqrt(((y_pred - y_true) ** 2).mean()))

    @torch.no_grad()
    def val_loss_with_head(self, head_state):
        tmp = copy.deepcopy(self.model).to(self.device)
        tmp.load_state_dict(head_state); tmp.eval()
        se, n = 0.0, 0
        for u, i, r in self.val_loader:
            u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
            y = tmp(u, i)
            se += torch.sum((y - r)**2).item()
            n += r.numel()
        return se / max(1, n)

    def _vec(self, params): return torch.cat([p.view(-1) for p in params])
    def _sq_l2(self, a, b): return 0.5 * torch.sum((self._vec(a) - self._vec(b))**2)

    def _train_core(self, head_state, lam_t, tau):
        head = copy.deepcopy(self.model).to(self.device)
        head.load_state_dict(head_state)
        self.model.train()
        for _ in range(self.local_epochs):
            for u, i, r in self.train_loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                self.opt.zero_grad()
                y = self.model(u, i)
                loss_mse = torch.mean((y - r)**2)
                sim = self._sq_l2(self.model.parameters(), head.parameters())
                stab = self._sq_l2(self.model.parameters(), self.old_model.parameters())
                loss = loss_mse + tau * (lam_t * sim + (1 - lam_t) * stab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
        self.old_model.load_state_dict(self.model.state_dict())

    def train_RF(self, head_state, lam_min, lam_max, t, T, tau=1.0):
        self.set_state(head_state)
        lam_t = lam_min + (lam_max - lam_min) * (t / max(1, T-1))
        self._train_core(head_state, lam_t, tau)
        return self.get_state(), len(self.train_loader.dataset)

    def train_RL(self, head_state, lam_min, lam_max, t, T, tau=1.0):
        lam_t = lam_min + (lam_max - lam_min) * (t / max(1, T-1))
        self._train_core(head_state, lam_t, tau)
        return self.get_state(), len(self.train_loader.dataset)
