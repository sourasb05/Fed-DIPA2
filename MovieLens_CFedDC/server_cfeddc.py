# server_cfeddc.py
import copy
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from client_cfeddc import CFedDCClient, RatingsDataset, MF  # make sure these exist in client_cfeddc.py


class CFedDCServer:
    """
    CFedDC server for MovieLens-style rating prediction.

    - Partitions TRAIN interactions into imbalanced client shards (done outside; pass client_datasets).
    - Per-round:
        * RF clients train from global state
        * RF clients clustered by parameter vectors (KMeans)
        * Cluster heads = FedAvg of members
        * RL clients select best cluster by lowest local val loss vs each head
        * RF -> RL model exchange inside chosen cluster
        * RL clients train
        * Recompute heads from RF+RL and aggregate to global (weighted by cluster data mass)
    - Metrics: RMSE on global val/test; mean client RMSE; optional JSON logging.
    """

    def __init__(
        self,
        # dataframes already encoded to integer ids (LabelEncoded) and split externally
        df_train,           # pandas DataFrame (columns: user_id, item_id, rating[, timestamp])
        df_val,             # pandas DataFrame
        df_test,            # pandas DataFrame
        # encoders sizes
        n_users: int,
        n_items: int,
        # federated settings
        num_clients: int = 20,
        seed: int = 0,
        rounds: int = 30,
        frac: float = 1.0,          # fraction of total clients participating each round
        n_clusters: int = 3,
        kappa: float = 0.8,         # fraction of RF clients participating
        delta: float = 0.5,         # fraction of RL clients participating
        # local training
        local_epochs: int = 1,
        batch_size: int = 2048,
        lr: float = 1e-3,
        # CFedDC regularization
        lam_min: float = 0.3,
        lam_max: float = 0.7,
        tau: float = 1.0,
        # logging
        output_json: str | None = None,
        device: str | None = None,
        # client shard constructor: list of (client_train_df, client_val_df)
        client_splits: list | None = None,
    ):
        """
        If you pass `client_splits`, it should be a list of tuples:
            [(cdf_train_0, cdf_val_0), (cdf_train_1, cdf_val_1), ...]
        where each cdf_* is a pandas DataFrame with (user_id, item_id, rating[, timestamp]).

        Otherwise, this server will build a single global client from df_train (for sanity), which
        isn't federated. In practice you should pass `client_splits`.
        """
        self.rng = np.random.RandomState(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rounds = rounds
        self.frac = float(frac)
        self.n_clusters = int(n_clusters)
        self.kappa = float(kappa)
        self.delta = float(delta)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.tau = float(tau)
        self.output_json = output_json

        self.n_users = int(n_users)
        self.n_items = int(n_items)

        # Build global eval loaders
        self.val_loader = DataLoader(
            RatingsDataset(df_val, user_encoder=None, item_encoder=None),  # dataset expects encoded ids already
            batch_size=8192, shuffle=False, num_workers=2, pin_memory=True
        )
        self.test_loader = DataLoader(
            RatingsDataset(df_test, user_encoder=None, item_encoder=None),
            batch_size=8192, shuffle=False, num_workers=2, pin_memory=True
        )

        # Build clients from precomputed splits
        self.clients = []
        if client_splits is None or len(client_splits) == 0:
            # Fallback: single-client (not really FL; for debugging only)
            train_ds = RatingsDataset(df_train, user_encoder=None, item_encoder=None)
            val_ds = RatingsDataset(df_val,   user_encoder=None, item_encoder=None)
            self.clients.append(
                CFedDCClient(
                    cid=0,
                    model_fn=lambda: MF(self.n_users, self.n_items, emb_dim=64),
                    train_ds=train_ds,
                    val_ds=val_ds,
                    device=self.device,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    local_epochs=self.local_epochs,
                )
            )
        else:
            for cid, (cdf_tr, cdf_va) in enumerate(client_splits):
                if len(cdf_tr) < 10 or len(cdf_va) < 5:
                    continue
                train_ds = RatingsDataset(cdf_tr, user_encoder=None, item_encoder=None)
                val_ds   = RatingsDataset(cdf_va, user_encoder=None, item_encoder=None)
                self.clients.append(
                    CFedDCClient(
                        cid=cid,
                        model_fn=lambda: MF(self.n_users, self.n_items, emb_dim=64),
                        train_ds=train_ds,
                        val_ds=val_ds,
                        device=self.device,
                        batch_size=self.batch_size,
                        lr=self.lr,
                        local_epochs=self.local_epochs,
                    )
                )

        if not self.clients:
            raise RuntimeError("No clients built. Provide valid client_splits with enough samples per client.")

        # Initialize global model/state
        self.global_model = MF(self.n_users, self.n_items, emb_dim=64).to(self.device)
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

        # Tracking (JSON)
        self.history = {
            "rounds": [],
            "rmse_val": [],
            "mean_client_rmse": [],
        }

        # For per-client best RMSE (lower is better). Initialize to +inf.
        self.best_client_rmse = {c.cid: float("inf") for c in self.clients}

    # -------------------- helpers --------------------

    @staticmethod
    def fedavg(states_and_sizes):
        """Weighted average of state_dicts."""
        total = sum(n for _, n in states_and_sizes) or 1
        keys = list(states_and_sizes[0][0].keys())
        agg = {k: 0 for k in keys}
        for state, n in states_and_sizes:
            w = n / total
            for k in keys:
                agg[k] = agg[k] + state[k].float() * w
        return agg

    @torch.no_grad()
    def rmse_global(self, state_dict, loader):
        """Compute RMSE for a given state on a loader."""
        model = MF(self.n_users, self.n_items, emb_dim=64).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        se, n = 0.0, 0
        for u, i, r in loader:
            u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
            y = model(u, i)
            se += torch.sum((y - r) ** 2).item()
            n += r.numel()
        return float(np.sqrt(se / max(1, n)))

    @staticmethod
    def split_resourceful(clients, top_frac=0.5):
        """
        RF = smallest set of clients, sorted by shard size desc, that cover >= top_frac of total samples.
        RL = rest.
        """
        sizes = [(c, len(c.train_loader.dataset)) for c in clients]
        sizes.sort(key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in sizes) or 1
        cum = 0
        rf = []
        for c, s in sizes:
            rf.append(c)
            cum += s
            if cum / total >= top_frac:
                break
        rl = [c for c in clients if c not in rf]
        return rf, rl

    # -------------------- main loop --------------------

    def run(self):
        print(f"Total clients: {len(self.clients)} | Rounds: {self.rounds} | clusters: {self.n_clusters}")

        for t in range(self.rounds):
            # Participation cap
            cap_total = max(1, int(self.frac * len(self.clients)))

            # RF/RL split by shard mass
            rf_all, rl_all = self.split_resourceful(self.clients, top_frac=0.5)

            # Sample RF/RL participants
            m_rf = min(max(1, int(self.kappa * len(rf_all))), len(rf_all))
            m_rl = min(max(0, int(self.delta * len(rl_all))), len(rl_all))
            # Enforce overall cap
            while m_rf + m_rl > cap_total and m_rl > 0:
                m_rl -= 1
            while m_rf + m_rl > cap_total and m_rf > 1:
                m_rf -= 1
            rf = random.sample(rf_all, m_rf) if m_rf > 0 else []
            rl = random.sample(rl_all, m_rl) if m_rl > 0 else []

            # ----- RF phase: train from global head to get fresh params -----
            for c in rf:
                c.train_RF(self.global_state, self.lam_min, self.lam_max, t, self.rounds, tau=self.tau)

            # ----- Cluster RF clients by parameter vectors -----
            if len(rf) >= self.n_clusters and self.n_clusters > 1:
                with torch.no_grad():
                    feats = torch.stack([c.get_param_vec() for c in rf]).numpy()
                try:
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
                    labels = kmeans.fit_predict(feats)
                    sil = float(silhouette_score(feats, labels))
                except Exception:
                    labels = np.zeros(len(rf), dtype=int)
                    sil = float("nan")
                clusters = [[] for _ in range(self.n_clusters)]
                for c, lab in zip(rf, labels):
                    clusters[int(lab)].append(c)
            else:
                clusters = [rf] if len(rf) > 0 else [[]]
                labels = [0] * len(rf)
                sil = float("nan")

            # Build initial cluster heads (RF only)
            cluster_heads = []
            for members in clusters:
                if members:
                    ss = [(c.get_state(), len(c.train_loader.dataset)) for c in members]
                    head = self.fedavg(ss)
                else:
                    head = copy.deepcopy(self.global_state)
                cluster_heads.append(head)

            # RF client id -> cluster id
            rf_cluster_id = {c.cid: lab for c, lab in zip(rf, labels)}

            # ----- RL selects best cluster by local val loss -----
            rl_best = {}
            for c in rl:
                losses = [c.val_loss_with_head(h) for h in cluster_heads]
                rl_best[c.cid] = int(np.argmin(losses)) if len(losses) > 0 else 0

            # ----- RF -> RL exchange inside chosen cluster (or start from head) -----
            for c in rl:
                k = rl_best[c.cid]
                rf_same = [r for r in rf if rf_cluster_id.get(r.cid, -1) == k]
                if rf_same:
                    donor = random.choice(rf_same)
                    c.load_exchange_from(donor.get_state())
                    c.set_state(donor.get_state())
                else:
                    c.set_state(cluster_heads[k])

            # ----- RL training -----
            for c in rl:
                k = rl_best[c.cid]
                c.train_RL(cluster_heads[k], self.lam_min, self.lam_max, t, self.rounds, tau=self.tau)

            # ----- Recompute heads from RF+RL within each cluster -----
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

            # ----- Global aggregation: weighted avg of heads by cluster mass -----
            W = sum(weights) or 1
            keys = list(new_heads[0].keys())
            self.global_state = {
                k: sum((head[k].float() * (w / W) for head, w in zip(new_heads, weights)))
                for k in keys
            }

            # ----- Metrics & logging -----
            rmse_val = self.rmse_global(self.global_state, self.val_loader)
            rmse_clients = [c.rmse_on_val() for c in self.clients]
            rmse_clients_mean = float(np.mean(rmse_clients)) if rmse_clients else float("nan")

            # Track best (lowest) client RMSEs
            for c, rmse in zip(self.clients, rmse_clients):
                if rmse < self.best_client_rmse[c.cid]:
                    self.best_client_rmse[c.cid] = rmse
            avg_best_client_rmse = float(np.mean(list(self.best_client_rmse.values())))

            print(
                f"[Round {t+1:03d}] RF:{len(rf)} RL:{len(rl)} "
                f"| clusters:{len(cluster_heads)} | sil:{sil:.3f} "
                f"| RMSE_val:{rmse_val:.4f} | mean_client_RMSE:{rmse_clients_mean:.4f} "
                f"| avg_best_client_RMSE:{avg_best_client_rmse:.4f}"
            )

            # Save to history (for JSON)
            self.history["rounds"].append(t + 1)
            self.history["rmse_val"].append(rmse_val)
            self.history["mean_client_rmse"].append(rmse_clients_mean)

        # Final test metric
        rmse_test = self.rmse_global(self.global_state, self.test_loader)
        print(f"\nFINAL Global Test RMSE: {rmse_test:.4f}")

        # Save JSON history (optional)
        if self.output_json:
            payload = {
                **self.history,
                "final_test_rmse": rmse_test,
                "avg_best_client_rmse": float(np.mean(list(self.best_client_rmse.values()))),
            }
            with open(self.output_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved results to {self.output_json}")
