# server_cfeddc.py
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score

from client_cfeddc import CFedDCClient, SimpleCNN
from utils_noniid import set_seed, imbalanced_dirichlet_partition, build_subsets

class CFedDCServer:
    def __init__(
        self,
        dataset_train, dataset_val, dataset_test,
        num_clients=50, alpha=0.3, seed=0,
        batch_size=64, lr=1e-3, local_epochs=1,
        rounds=30, frac=1.0, n_clusters=3, kappa=0.8, delta=0.5,
        lam_min=0.3, lam_max=0.7, tau=1.0, device=None
    ):
        set_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rounds = rounds
        self.frac = frac
        self.n_clusters = n_clusters
        self.kappa = kappa
        self.delta = delta
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.tau = tau


        # Partition train set non-IID
        client_indices = imbalanced_dirichlet_partition(targets=dataset_train.targets,        
                                                        n_clients=num_clients,
                                                        alpha=0.3,                         # label non-IID
                                                        min_size=20,
                                                        seed=0,
                                                        # Option A: log-normal sizes (default)
                                                        lognormal_params=(3.0, 1.2),
                                                        )
        client_train_subsets = build_subsets(dataset_train, client_indices)

        # Each client: split its shard into local train/val (80/20)
        self.clients = []
        for cid, subset in enumerate(client_train_subsets):
            n = len(subset)
            print(n)
            if n < 50:
                continue
            n_val_local = max(20, int(0.2 * n))
            n_train_local = n - n_val_local
            tr_local, va_local = random_split(
                subset, [n_train_local, n_val_local],
                generator=torch.Generator().manual_seed(seed + cid)
            )
            self.clients.append(
                CFedDCClient(
                    cid=cid,
                    model_fn=lambda: SimpleCNN(num_classes=10),
                    train_subset=tr_local,
                    val_subset=va_local,
                    device=self.device,
                    batch_size=batch_size,
                    lr=lr,
                    local_epochs=local_epochs,
                )
            )

        if not self.clients:
            raise RuntimeError("No clients with enough data after partitioning.")
        
        self.best_client_acc = {c.cid: 0.0 for c in self.clients}

        # Global eval loaders
        self.val_loader  = DataLoader(dataset_val,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

        # Init global model/state
        self.global_model = SimpleCNN(10).to(self.device)
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    # ---- helpers ----
    @staticmethod
    def fedavg(states_and_sizes):
        total = sum(n for _, n in states_and_sizes) or 1
        keys = list(states_and_sizes[0][0].keys())
        agg = {k: 0 for k in keys}
        for state, n in states_and_sizes:
            w = n / total
            for k in keys:
                agg[k] = agg[k] + state[k].float() * w
        return agg

    @torch.no_grad()
    def f1_global(self, state_dict, loader):
        model = SimpleCNN(10).to(self.device)
        model.load_state_dict(state_dict); model.eval()
        y_true, y_pred = [], []
        for x, y in loader:
            x = x.to(self.device)
            preds = torch.argmax(model(x), dim=1).cpu()
            y_pred.append(preds); y_true.append(y)
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        return float(f1_score(y_true, y_pred, average="macro"))

    @staticmethod
    def split_resourceful(clients, top_frac=0.5):
        """Split clients into RF (top by data mass until >= top_frac of total) and RL (rest)."""
        sizes = [(c, len(c.train_loader.dataset)) for c in clients]
        sizes.sort(key=lambda x: x[1], reverse=True)
        cum = 0
        total = sum(s for _, s in sizes) or 1
        rf = []
        for c, s in sizes:
            rf.append(c); cum += s
            if cum / total >= top_frac:
                break
        rl = [c for c in clients if c not in rf]
        return rf, rl

    def run(self):
        print(f"Total clients: {len(self.clients)} | Rounds: {self.rounds} | clusters: {self.n_clusters}")

        for t in range(self.rounds):
            # fraction of clients (cap)
            cap_total = max(1, int(self.frac * len(self.clients)))

            # RF/RL split by data mass
            rf_all, rl_all = self.split_resourceful(self.clients, top_frac=0.5)

            # pick RF/RL participants by kappa/delta
            m_rf = min(max(1, int(self.kappa * len(rf_all))), len(rf_all))
            m_rl = min(max(0, int(self.delta * len(rl_all))), len(rl_all))
            # enforce overall cap
            while m_rf + m_rl > cap_total and m_rl > 0: m_rl -= 1
            while m_rf + m_rl > cap_total and m_rf > 1: m_rf -= 1

            rf = random.sample(rf_all, m_rf) if m_rf > 0 else []
            rl = random.sample(rl_all, m_rl) if m_rl > 0 else []

            # RF train w.r.t global (warm-up to get fresh params for clustering)
            for c in rf:
                c.train_RF(self.global_state, self.lam_min, self.lam_max, t, self.rounds, tau=self.tau)

            # Cluster RF clients (KMeans on param vectors). If too few RF, single cluster.
            if len(rf) >= self.n_clusters:
                with torch.no_grad():
                    feats = torch.stack([c.get_param_vec() for c in rf]).numpy()
                try:
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=0)
                    labels = kmeans.fit_predict(feats)
                    sil = float(silhouette_score(feats, labels))
                except Exception:
                    labels = np.zeros(len(rf), dtype=int)
                    sil = float("nan")
                clusters = [[] for _ in range(self.n_clusters)]
                for c, lab in zip(rf, labels):
                    clusters[lab].append(c)
            else:
                clusters = [rf]  # single cluster
                labels = [0] * len(rf)
                sil = float("nan")

            # Build initial cluster heads from RF (FedAvg)
            cluster_heads = []
            for members in clusters:
                if members:
                    ss = [(c.get_state(), len(c.train_loader.dataset)) for c in members]
                    head = self.fedavg(ss)
                else:
                    head = copy.deepcopy(self.global_state)
                cluster_heads.append(head)

            # Map RF client id → cluster id
            rf_cluster_id = {}
            for c, lab in zip(rf, labels):
                rf_cluster_id[c.cid] = int(lab)

            # RL assigns to best head (lowest local val CE)
            rl_best = {}
            for c in rl:
                losses = [c.val_loss_with_head(h) for h in cluster_heads]
                rl_best[c.cid] = int(np.argmin(losses)) if losses else 0

            # RF → RL exchange inside cluster (copy donor weights if available)
            for c in rl:
                k = rl_best[c.cid]
                rf_same = [r for r in rf if rf_cluster_id.get(r.cid, -1) == k]
                if rf_same:
                    donor = random.choice(rf_same)
                    c.load_exchange_from(donor.get_state())
                    c.set_state(donor.get_state())
                else:
                    c.set_state(cluster_heads[k])

            # RL train w.r.t chosen head
            for c in rl:
                k = rl_best[c.cid]
                c.train_RL(cluster_heads[k], self.lam_min, self.lam_max, t, self.rounds, tau=self.tau)

            # Recompute heads from RF+RL members
            full_clusters = [[] for _ in range(len(cluster_heads))]
            for c in rf:
                full_clusters[rf_cluster_id.get(c.cid, 0)].append(c)
            for c in rl:
                full_clusters[rl_best[c.cid]].append(c)

            new_heads = []
            weights = []
            for members in full_clusters:
                if members:
                    ss = [(c.get_state(), len(c.train_loader.dataset)) for c in members]
                    new_heads.append(self.fedavg(ss))
                    weights.append(sum(len(c.train_loader.dataset) for c in members))
                else:
                    new_heads.append(copy.deepcopy(self.global_state))
                    weights.append(0)

            # Global = weighted average of cluster heads by cluster data mass
            W = sum(weights) or 1
            keys = list(new_heads[0].keys())
            self.global_state = {
                k: sum((head[k].float() * (w / W) for head, w in zip(new_heads, weights)))
                for k in keys
            }

            # Logging: global macro-F1 on val + mean client local-val F1
            f1_val = self.f1_global(self.global_state, self.val_loader)
            f1_clients = [c.f1_on_val() for c in self.clients]
            f1_clients_mean = float(np.mean(f1_clients)) if f1_clients else 0.0
            print(f"[Round {t+1:03d}] RF:{len(rf)} RL:{len(rl)} | clusters:{len(cluster_heads)} | sil:{sil:.3f} | F1_val:{f1_val:.4f} | mean_client_F1:{f1_clients_mean:.4f}")
            # Track per-client accuracy
            for c in self.clients:
                acc = c.acc_on_val()
                if acc > self.best_client_acc[c.cid]:
                    self.best_client_acc[c.cid] = acc

            # Average best accuracy across clients
            avg_best_acc = float(np.mean(list(self.best_client_acc.values())))
            
            print(f"[Round {t+1:03d}] RF:{len(rf)} RL:{len(rl)} | clusters:{len(cluster_heads)} | sil:{sil:.3f} | F1_val:{f1_val:.4f} | mean_client_F1:{f1_clients_mean:.4f} | Avg best client acc:{avg_best_acc:.4f}")

            # Final test
        # f1_test = self.f1_global(self.global_state, self.test_loader)
        # print(f"\nFINAL Global Test F1(macro): {f1_test:.4f}")
