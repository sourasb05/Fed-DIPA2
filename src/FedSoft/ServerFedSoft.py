import os
import copy
import json
import wandb
import torch
import random
import datetime
import numpy as np
from tqdm import trange, tqdm

from src.utils.results_utils import InformativenessMetrics, CalculateMetrics
from src.FedSoft.Client_FedSoft import UserFedSoft   # <— import the client below


class FedSoftServer:
    """
    FedSoft server:
      - Maintains S centers c_s (one model per cluster).
      - Every τ rounds: asks all clients to estimate u_{k,s} via per-sample loss argmin.
      - Each round:
          * compute per-cluster sampling probs v_{t,k,s} ∝ u_{k,s} * n_k   (Eq. 6)
          * sample K clients per cluster (without duplication across clusters if possible)
          * selected clients do ONE proximal local update with all centers in the regularizer
          * aggregate each center s with the local models of clients selected for s
    """
    def __init__(self, device, args, exp_no, current_directory):
        self.device = device
        self.args = args
        self.exp_no = exp_no
        self.current_directory = current_directory

        # Training/meta hyperparams
        self.num_glob_iters   = args.num_global_iters
        self.local_iters      = args.local_iters
        self.batch_size       = args.batch_size
        self.alpha            = args.alpha            # client optimizer LR
        self.eta              = args.eta
        self.algorithm        = "FedSoft"

        # FedSoft-specific
        self.S                = args.num_clusters     # number of clusters (centers)
        self.tau              = getattr(args, "tau", 2)           # estimation interval
        self.K                = getattr(args, "cluster_k", 10)    # clients per cluster per round
        self.lamda_prox       = getattr(args, "lamda_prox", 1.0)  # λ in proximal term
        self.sigma_smoother   = getattr(args, "sigma", 1e-4)      # σ in paper

        # Users / country split (kept from your FedProx code)
        if args.country == "japan":
            self.user_ids = args.user_ids[0]
        elif args.country == "uk":
            self.user_ids = args.user_ids[1]
        elif args.country == "both":
            self.user_ids = args.user_ids[3]
        else:
            self.user_ids = args.user_ids[2]

        self.users = []
        self.total_train_samples = 0

        # W&B
        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(
            project="DIPA2",
            name=f"FedSoft_{date_and_time}_{len(self.user_ids)}",
            mode=None if args.wandb else "disabled"
        )

        # Build clients
        for uid in trange(len(self.user_ids), desc="Data distribution to clients"):
            u = UserFedSoft(
                device=self.device,
                args=args,
                id=int(self.user_ids[uid]),
                exp_no=self.exp_no,
                current_directory=self.current_directory,
                wandb=self.wandb,
                num_clusters=self.S,
                lamda_prox=self.lamda_prox,
                sigma=self.sigma_smoother
            )
            if u.valid:
                self.users.append(u)
                self.total_train_samples += u.train_samples

        self.total_users = len(self.users)
        if self.total_users == 0:
            raise RuntimeError("No valid users constructed for FedSoft.")

        # Initialize S centers (start as zeroed copy of a model)
        base = copy.deepcopy(self.users[0].local_model).to(self.device)
        for p in base.parameters():
            p.data.zero_()
        self.centers = [copy.deepcopy(base) for _ in range(self.S)]  # c_s
        self.center_min_loss = [float("inf")] * self.S

        # Book-keeping
        self.minimum_test_loss = float('inf')

        print(f"[FedSoft] total users: {self.total_users}, clusters (S): {self.S}, "
              f"K per cluster: {self.K}, tau: {self.tau}, λ: {self.lamda_prox}, σ: {self.sigma_smoother}")

    def __del__(self):
        try:
            self.wandb.finish()
        except Exception:
            pass

    # -----------------------------
    # Helper: copy parameters/model
    # -----------------------------
    def _copy_model_params_from_to(self, src_model, dst_model):
        for p_dst, p_src in zip(dst_model.parameters(), src_model.parameters()):
            p_dst.data.copy_(p_src.data)

    def _average_models(self, models, weights=None):
        """
        Return a new model as (weighted) average of `models`.
        """
        out = copy.deepcopy(models[0])
        with torch.no_grad():
            for p in out.parameters():
                p.data.zero_()

            if weights is None:
                w = [1.0 / len(models)] * len(models)
            else:
                # normalize
                s = sum(weights)
                if s <= 0:
                    w = [1.0 / len(models)] * len(models)
                else:
                    w = [wi / s for wi in weights]

            for mi, m in enumerate(models):
                for p_out, p_m in zip(out.parameters(), m.parameters()):
                    p_out.data.add_(p_m.data, alpha=w[mi])
        return out

    # -------------------------------------------------
    # Step 1 (every τ rounds): estimate u_{k,s} on all clients
    # -------------------------------------------------
    def _estimate_importance_weights_all(self):
        # broadcast centers to all clients; each computes per-sample argmin loss counts
        for u in tqdm(self.users, desc="[FedSoft] Estimating importance weights u_{k,s}"):
            u.set_centers(self.centers)
            u.estimate_importance_weights()  # fills u.u_ks (length S), also exposes counts & n_k to server if needed

    # -------------------------------------------------
    # Step 2: compute v_{t,k,s} ∝ u_{k,s} * n_k (Eq. 6 in the paper)
    # -------------------------------------------------
    def _compute_v_probs(self):
        # For each cluster s, build probs over clients
        v_probs = []
        for s in range(self.S):
            numer = []
            for u in self.users:
                numer.append(u.u_ks[s] * u.train_samples)  # u_{k,s} * n_k
            total = sum(numer)
            if total <= 0:
                # fall back to uniform
                probs = [1.0 / self.total_users] * self.total_users
            else:
                probs = [x / total for x in numer]
            v_probs.append(probs)
        return v_probs  # shape: S x total_users

    # -------------------------------------------------
    # Step 3: sample K clients per cluster, without duplicate if possible
    # -------------------------------------------------
    def _sample_clients_per_cluster(self, v_probs):
        rng = np.random.default_rng(seed=None)
        chosen_indices_per_s = []
        already = set()
        for s in range(self.S):
            probs = np.array(v_probs[s], dtype=float)
            # set prob ~0 for already chosen to reduce duplicates
            if len(already) > 0:
                mask = np.ones_like(probs, dtype=bool)
                mask[list(already)] = False
                if mask.sum() >= self.K:
                    # sample from available
                    p = probs * mask
                    if p.sum() == 0:
                        p = mask.astype(float) / mask.sum()
                    else:
                        p = p / p.sum()
                    idx = rng.choice(self.total_users, size=self.K, replace=False, p=p)
                else:
                    # take as many as we can; then sample rest allowing duplicates
                    p = probs.copy()
                    p = p / p.sum() if p.sum() > 0 else np.ones_like(probs)/len(probs)
                    idx = rng.choice(self.total_users, size=self.K, replace=False, p=p)
            else:
                p = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
                idx = rng.choice(self.total_users, size=self.K, replace=False, p=p)

            chosen_indices_per_s.append(idx.tolist())
            already.update(idx.tolist())
        return chosen_indices_per_s  # list of length S; each a list of user indices

    # -------------------------------------------------
    # Step 4: selected clients do one proximal local update, send back their local model
    # -------------------------------------------------
    def _clients_local_prox_updates(self, chosen_indices_per_s):
        """
        Returns:
          per_cluster_models: list length S; each is a list of client models selected for that cluster
          per_cluster_weights: list length S; weights (v_{t,k,s} * n_k) or just (u_{k,s} * n_k) (we weight in avg)
        """
        # Let the selected sets overlap if sampling produced overlap. Each client runs ONE local solve h_k
        # (prox objective with all centers). We'll reuse the single updated local model in all clusters it's chosen for.
        updated_models = [None] * self.total_users  # cache by user index
        for idx, u in enumerate(self.users):
            u.set_centers(self.centers)  # ensure up-to-date

        # Determine the set of all selected indices (union over clusters)
        selected_union = set()
        for s in range(self.S):
            selected_union.update(chosen_indices_per_s[s])
        selected_union = sorted(list(selected_union))

        # Run local updates for the union
        for ui in tqdm(selected_union, desc="[FedSoft] Local proximal updates"):
            u = self.users[ui]
            updated_models[ui] = u.local_prox_update()  # returns a deepcopy(model) with updated params

        # Gather per-cluster model lists + weights (use u_{k,s} * n_k as aggregation weights; server will normalize)
        per_cluster_models = [[] for _ in range(self.S)]
        per_cluster_weights = [[] for _ in range(self.S)]

        for s in range(self.S):
            for ui in chosen_indices_per_s[s]:
                u = self.users[ui]
                per_cluster_models[s].append(updated_models[ui])
                per_cluster_weights[s].append(u.u_ks[s] * u.train_samples)  # proportional to v_{t,k,s} numerator

        return per_cluster_models, per_cluster_weights

    # -------------------------------------------------
    # Step 5: aggregate each center s by weighted average of received local models
    # -------------------------------------------------
    def _aggregate_centers(self, per_cluster_models, per_cluster_weights):
        for s in range(self.S):
            if len(per_cluster_models[s]) == 0:
                continue
            new_cs = self._average_models(per_cluster_models[s], per_cluster_weights[s])
            self._copy_model_params_from_to(new_cs, self.centers[s])

    # --------------------
    # (Optional) save centers
    # --------------------
    def _save_centers(self, t):
        base_dir = os.path.join(self.current_directory, "models", self.algorithm, "centers",
                                f"_GE_{self.num_glob_iters}_LE_{self.local_iters}")
        os.makedirs(base_dir, exist_ok=True)
        for s in range(self.S):
            ckpt = {'GR': t, 'center_index': s, 'model_state_dict': self.centers[s].state_dict()}
            torch.save(ckpt, os.path.join(base_dir, f"center_{s}_GR{t}.pt"))

    # --------------------
    # Train (main loop)
    # --------------------
    def train(self):
        for t in trange(self.num_glob_iters, desc="[FedSoft] Global Rounds"):
            # Every τ rounds: recompute the u_{k,s}
            if t % self.tau == 0:
                self._estimate_importance_weights_all()

            # Compute sampling probabilities v_{t,k,s} (proportional to u_{k,s} n_k)
            v_probs = self._compute_v_probs()

            # Sample clients for each cluster
            chosen = self._sample_clients_per_cluster(v_probs)

            # Run one local proximal solve per selected client; collect models for each cluster
            per_cluster_models, per_cluster_weights = self._clients_local_prox_updates(chosen)

            # Aggregate to update centers
            self._aggregate_centers(per_cluster_models, per_cluster_weights)

            # (Optional) lightweight logging
            self.wandb.log({"round": t})

            # (Optional) evaluate personalized/local vs centers here if you want (left minimal to avoid extra runtime)
            # You can plug your existing evaluate_* methods by pointing them to self.centers or user.personalized_model

            # Save checkpoints occasionally
            if (t == self.num_glob_iters - 1) or (t % max(1, self.num_glob_iters // 5) == 0):
                self._save_centers(t)

        # Optionally dump per-user results (omitted for brevity, follow your FedProx save_results if needed)
        print("[FedSoft] Training complete.")
