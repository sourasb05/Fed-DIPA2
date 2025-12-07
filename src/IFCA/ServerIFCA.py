import os
import copy
import json
import wandb
import torch
import numpy as np
import datetime
from tqdm import trange, tqdm

from src.IFCA.UserIFCA import UserIFCA  
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics


class ServerIFCA:
    """
    Iterative Federated Clustering Algorithm (IFCA) server.
    Keeps K cluster models, assigns users to clusters each round,
    performs per-cluster aggregation, and evaluates with users' current
    cluster assignments.
    """
    def __init__(self, device, args, exp_no, current_directory):
        self.device = device
        self.args = args
        self.exp_no = exp_no
        self.algorithm = "IFCA"

        if args.model_name == "openai_ViT-L/14@336px":
            self.model_name = "ViT-L_14_336px"
        else:
            self.model_name = args.model_name

        self.current_directory = current_directory

        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.num_clusters = int(args.num_teams)
        assert self.num_clusters >= 2, "IFCA requires num_clusters >= 2"

        # Country-specific user selection (kept from your template)
        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[0]
        elif args.country == "uk":
            self.user_ids = args.user_ids[1]
        elif args.country == "both":
            self.user_ids = args.user_ids[3]
        else:
            self.user_ids = args.user_ids[2]

        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")

        # bookkeeping
        self.users = []
        self.selected_users = []
        self.assignments = {}  # user_id -> cluster_id

        self.global_test_metric = []
        self.global_test_loss = []
        self.global_test_distance = []
        self.global_test_mae = []

        self.global_train_metric = []
        self.global_train_loss = []
        self.global_train_distance = []
        self.global_train_mae = []

        self.data_frac = []
        self.minimum_val_cmae = float("inf")

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(
            project="DIPA2",
            name=f"IFCA_{date_and_time}_{self.total_users}u_{self.num_clusters}c",
            mode=None if args.wandb else "disabled",
        )

        # Build users (like FedProx) — but use UserIFCA
        self.total_train_samples = 0
        for i in trange(self.total_users, desc="Data distribution to clients"):
            uid = int(self.user_ids[i])
            user = UserIFCA(device, args, uid, exp_no, current_directory, self.wandb)
            if user.valid:
                self.users.append(user)
                self.total_train_samples += user.train_samples

        self.total_users = len(self.users)
        self.num_users = int(self.total_users * args.users_frac)  # selected per round

        # Data fractions and init cluster models
        for user in self.users:
            self.data_frac.append(user.train_samples / self.total_train_samples)

        # Initialize K cluster models by cloning user[0]'s local model
        # (zero-initialize weights to mimic your FedProx start)
        base_model = copy.deepcopy(self.users[0].local_model)
        for p in base_model.parameters():
            p.data.zero_()

        self.cluster_models = []
        for k in range(self.num_clusters):
            m = copy.deepcopy(base_model).to(self.device)
            self.cluster_models.append(m)

        print("Finished creating IFCA server with K =", self.num_clusters)

    def __del__(self):
        try:
            self.wandb.finish()
        except Exception:
            pass

    # --------- Utils ----------
    def save_all_cluster_models(self, glob_iter, tag="checkpoint"):
        model_path = os.path.join(
            self.current_directory, "models", self.algorithm, "global_models",
            f"_GE_{self.num_glob_iters}_LE_{self.local_iters}"
        )
        os.makedirs(model_path, exist_ok=True)
        for k, model in enumerate(self.cluster_models):
            checkpoint = {"GR": glob_iter, "cluster": k, "model_state_dict": model.state_dict()}
            torch.save(checkpoint, os.path.join(model_path, f"{tag}_cluster{k}_GR{glob_iter}.pt"))

    def select_users(self, round_idx, subset_users):
        if subset_users >= len(self.users):
            return list(self.users)
        np.random.seed(round_idx)
        choice = np.random.choice(self.users, subset_users, replace=False)
        return list(choice)

    # --------- IFCA core steps ----------
    def e_step_assign_clusters(self, users, cluster_models, mode="train"):
        """
        E-step: each user chooses best-fit cluster by scoring all K models.
        mode in {"train","eval"} only controls logs; selection logic identical.
        """
        assignments = {}
        for user in tqdm(users, desc=f"IFCA E-step ({mode})"):
            k_star, losses = user.select_cluster(cluster_models)
            assignments[user.id] = k_star
            # Optional logging
            self.wandb.log({f"assign/{mode}/user_{user.id}": int(k_star)})
        return assignments

    def m_step_local_train(self, users, assignments, cluster_models, glob_iter):
        """
        M-step: users train locally starting from their assigned cluster model.
        """
        for user in tqdm(users, desc="IFCA M-step: local training"):
            k = assignments[user.id]
            user.train_on_cluster(glob_iter, cluster_models[k])

    def aggregate_per_cluster(self, users, assignments):
        """
        Aggregate updated client models per cluster (FedAvg-style),
        weighted by train_samples. If a cluster has no users this round,
        we keep its previous model.
        """
        # Prepare buffers per cluster
        cluster_user_lists = [[] for _ in range(self.num_clusters)]
        for u in users:
            cluster_user_lists[assignments[u.id]].append(u)

        for k in range(self.num_clusters):
            assigned = cluster_user_lists[k]
            if len(assigned) == 0:
                # No update for this cluster this round
                continue

            # zero-out cluster model
            for p in self.cluster_models[k].parameters():
                p.data = torch.zeros_like(p.data)

            total = sum(u.train_samples for u in assigned)
            # weighted sum
            for u in assigned:
                ratio = u.train_samples / total
                for p_c, p_u in zip(self.cluster_models[k].parameters(), u.get_parameters()):
                    p_c.data = p_c.data + p_u.data.clone() * ratio

    # --------- Evaluation (kept close to your FedProx style) ----------
    def evaluate_local(self, t):
        # Local models: just call each user's local test (independent of cluster)
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        val_avg_f1 = 0.0

        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0

        denom = max(1, len(self.selected_users))
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_local_model_val()
            test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, _ = c.test_local_model_test()

            val_avg_mae += (1 / denom) * info_mae
            val_avg_cmae += (1 / denom) * info_cmae
            val_avg_f1 += (1 / denom) * info_f1

            test_avg_mae += (1 / denom) * test_info_mae
            test_avg_cmae += (1 / denom) * test_info_cmae
            test_avg_f1 += (1 / denom) * test_info_f1

        print(f"\033[92m\n Global round {t} : Local val cmae {val_avg_cmae:.4f}  Local val mae {val_avg_mae:.4f} \033[0m")
        print(f"\033[93m\n Global round {t} : Local test cmae {test_avg_cmae:.4f} Local test mae {test_avg_mae:.4f} \033[0m")

    def evaluate_global(self, t):
        """
        Evaluate with current cluster models. We re-assign ALL users (E-step in eval mode),
        then score each user on their assigned cluster model (val + test).
        """
        # Eval-time reassignment (so every user has a k)
        self.assignments = self.e_step_assign_clusters(self.users, self.cluster_models, mode="eval")

        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        val_avg_f1 = 0.0

        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0

        denom = max(1, len(self.users))
        for c in self.users:
            k = self.assignments[c.id]
            # validation
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_global_model_val(self.cluster_models[k])
            # test
            t_prec, t_rec, t_f1, t_cmae, t_mae, _ = c.test_global_model_test(self.cluster_models[k])

            val_avg_mae += (1 / denom) * info_mae
            val_avg_cmae += (1 / denom) * info_cmae
            val_avg_f1 += (1 / denom) * info_f1

            test_avg_mae += (1 / denom) * t_mae
            test_avg_cmae += (1 / denom) * t_cmae
            test_avg_f1 += (1 / denom) * t_f1

        print(f"\n Global round {t} : Global val f1: {val_avg_f1:.4f}  Global val cmae: {val_avg_cmae:.4f}  Global val mae: {val_avg_mae:.4f}")
        print(f" Global round {t} : Global test f1: {test_avg_f1:.4f} Global test cmae: {test_avg_cmae:.4f} Global test mae: {test_avg_mae:.4f}\n")

        # save the best by validation cMAE (lower is better)
        if val_avg_cmae < self.minimum_val_cmae:
            self.minimum_val_cmae = val_avg_cmae
            self.save_all_cluster_models(t, tag="best")

    # --------- Persist results (re-use clients’ dicts) ----------
    def convert_numpy(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_results(self):
        for user in self.users:
            val_dict = user.val_round_result_dict
            test_dict = user.test_round_result_dict
            global_val_dict = user.val_global_round_result_dict
            global_test_dict = user.test_global_round_result_dict

            user_id = str(user.id)
            base = f"results/exp_{self.exp_no}_model_name_{self.model_name}_{self.algorithm}"

            paths = {
                "local_val": f"{base}/local_val/user_{user_id}_val_round_results.json",
                "local_test": f"{base}/local_test/user_{user_id}_test_round_results.json",
                "global_val": f"{base}/global_val/user_{user_id}_val_round_results.json",
                "global_test": f"{base}/global_test/user_{user_id}_test_round_results.json",
            }

            os.makedirs(os.path.dirname(paths["local_val"]), exist_ok=True)
            os.makedirs(os.path.dirname(paths["local_test"]), exist_ok=True)
            os.makedirs(os.path.dirname(paths["global_val"]), exist_ok=True)
            os.makedirs(os.path.dirname(paths["global_test"]), exist_ok=True)

            with open(paths["local_val"], "w") as f:
                json.dump({"User": user_id, "validation_results": val_dict}, f, indent=2, default=self.convert_numpy)
            with open(paths["local_test"], "w") as f:
                json.dump({"User": user_id, "validation_results": test_dict}, f, indent=2, default=self.convert_numpy)
            with open(paths["global_val"], "w") as f:
                json.dump({"User": user_id, "validation_results": global_val_dict}, f, indent=2, default=self.convert_numpy)
            with open(paths["global_test"], "w") as f:
                json.dump({"User": user_id, "validation_results": global_test_dict}, f, indent=2, default=self.convert_numpy)

    # --------- Main training loop ----------
    def train(self):
        for glob_iter in trange(self.num_glob_iters, desc="Global Rounds (IFCA)"):
            # choose participants
            self.selected_users = self.select_users(glob_iter, self.num_users)
            sel_ids = [u.id for u in self.selected_users]
            print(f"Exp {self.exp_no}: round {glob_iter} selected users: {sel_ids}")

            # E-step (train-time) — assign clusters by scoring K models
            self.assignments = self.e_step_assign_clusters(self.selected_users, self.cluster_models, mode="train")

            # M-step — local training on assigned cluster model
            self.m_step_local_train(self.selected_users, self.assignments, self.cluster_models, glob_iter)

            # Aggregation per cluster
            self.aggregate_per_cluster(self.selected_users, self.assignments)

            # Evaluate
            self.evaluate_local(glob_iter)
            self.evaluate_global(glob_iter)

        # Save last-round checkpoints of all clusters
        self.save_all_cluster_models(self.num_glob_iters - 1, tag="final")
        # Persist per-user metric histories
        self.save_results()
