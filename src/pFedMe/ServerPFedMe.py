import os
import copy
import json
import wandb
import torch
import numpy as np
import datetime
from tqdm import trange, tqdm

from src.pFedMe.UserPFedMe import UserPFedMe
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics


class ServerPFedMe:
    """
    pFedMe server:
      - keeps one global model w
      - clients solve local Moreau-regularized subproblems to get personalized x_i
      - server meta-update: w <- w + eta * lambda * (x_bar - w)
    """

    def __init__(self, device, args, exp_no, current_directory):
        self.device = device
        self.args = args
        self.exp_no = exp_no
        self.current_directory = current_directory

        self.algorithm = "pFedMe"
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size

        # inner lr (client) and outer/meta lr (server)
        self.inner_lr = args.alpha            # client step size
        self.meta_lr = args.eta              # server/meta step
        self.lamda = args.lamda_sim_sta      # Moreau/Prox regularizer λ

        # Country-based user selection (kept from your template)
        self.country = args.country
        if args.model_name == "openai_ViT-L/14@336px":
            self.model_name = "ViT-L_14_336px"
        else:
            self.model_name = args.model_name

 
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
        self.data_frac = []
        self.total_train_samples = 0

        self.minimum_val_cmae = float("inf")

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(
            project="DIPA2",
            name=f"pFedMe_{date_and_time}_{self.total_users}u",
            mode=None if args.wandb else "disabled",
        )

        # Build clients
        for i in trange(self.total_users, desc="Data distribution to clients"):
            uid = int(self.user_ids[i])
            user = UserPFedMe(device, args, uid, exp_no, current_directory, self.wandb)
            if user.valid:
                self.users.append(user)
                self.total_train_samples += user.train_samples

        self.total_users = len(self.users)
        self.num_users = int(self.total_users * args.users_frac)

        # Data fractions
        for user in self.users:
            self.data_frac.append(user.train_samples / self.total_train_samples)

        # Global init model from a user model (keep normal initialization)
        self.global_model = copy.deepcopy(self.users[0].local_model).to(self.device)
        print("Finished creating pFedMe server.")

    def __del__(self):
        try:
            self.wandb.finish()
        except Exception:
            pass

    # ---------- selection ----------
    def select_users(self, round_idx, subset_users):
        if subset_users >= len(self.users):
            return list(self.users)
        np.random.seed(round_idx)
        choice = np.random.choice(self.users, subset_users, replace=False)
        return list(choice)

    # ---------- server meta update: w <- w + eta*lambda*(x_bar - w) ----------
    @torch.no_grad()
    def meta_update_from_personalized(self, personalized_param_lists, weights):
        """
        personalized_param_lists: list of lists of tensors (each client's x_i after local training)
        weights: list of scalars summing to 1 (train-sample weighting)
        """
        weights = [float(w) for w in weights]
        for pi, gparam in enumerate(self.global_model.parameters()):
            # weighted average of personalized params for this tensor index
            avg = torch.zeros_like(gparam)
            for client_params, w in zip(personalized_param_lists, weights):
                avg += w * client_params[pi]
            # pFedMe meta step
            gparam += self.meta_lr * self.lamda * (avg - gparam)

    # ---------- evaluation ----------
    def evaluate_local(self, t):
        # Evaluate each user's personalized (local) model
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        val_avg_f1 = 0.0

        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0

        denom = max(1, len(self.users))
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_local_model_val()
            t_prec, t_rec, t_f1, t_cmae, t_mae, _ = c.test_local_model_test()

            val_avg_mae += (1/denom) * info_mae
            val_avg_cmae += (1/denom) * info_cmae
            val_avg_f1 += (1/denom) * info_f1

            test_avg_mae += (1/denom) * t_mae
            test_avg_cmae += (1/denom) * t_cmae
            test_avg_f1 += (1/denom) * t_f1

        print(f"\033[92m\n Global round {t} : Local val cMAE {val_avg_cmae:.4f}  Local val MAE {val_avg_mae:.4f} \033[0m")
        print(f"\033[93m\n Global round {t} : Local test cMAE {test_avg_cmae:.4f} Local test MAE {test_avg_mae:.4f} \033[0m")

    def evaluate_global(self, t):
        """
        Optional: evaluate the *global init* by copying it into an eval model and
        scoring on users' val/test sets (no personalization). This is not the main
        pFedMe metric but useful to monitor.
        """
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        val_avg_f1 = 0.0

        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0

        denom = max(1, len(self.users))
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_global_model_val(self.global_model)
            t_prec, t_rec, t_f1, t_cmae, t_mae, _ = c.test_global_model_test(self.global_model)

            val_avg_mae += (1/denom) * info_mae
            val_avg_cmae += (1/denom) * info_cmae
            val_avg_f1 += (1/denom) * info_f1

            test_avg_mae += (1/denom) * t_mae
            test_avg_cmae += (1/denom) * t_cmae
            test_avg_f1 += (1/denom) * t_f1

        print(f"\n Global round {t} : Global(val) f1 {val_avg_f1:.4f} cMAE {val_avg_cmae:.4f} MAE {val_avg_mae:.4f}")
        print(f" Global round {t} : Global(test) f1 {test_avg_f1:.4f} cMAE {test_avg_cmae:.4f} MAE {test_avg_mae:.4f}\n")

        # track best by local val cMAE (pFedMe target is personalization)
        if val_avg_cmae < self.minimum_val_cmae:
            self.minimum_val_cmae = val_avg_cmae
            self.save_global_model(t, tag="best")

    # ---------- saving ----------
    def save_global_model(self, glob_iter, tag="checkpoint"):
        model_path = os.path.join(
            self.current_directory,
            "models", self.algorithm, "global_model",
            f"_GE_{self.num_glob_iters}_LE_{self.local_iters}",
        )
        os.makedirs(model_path, exist_ok=True)
        checkpoint = {
            "GR": glob_iter,
            "model_state_dict": self.global_model.state_dict(),
            "best_val_cmae": self.minimum_val_cmae,
        }
        torch.save(checkpoint, os.path.join(model_path, f"{tag}_server_checkpoint_GR{glob_iter}.pt"))

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

    # ---------- main loop ----------
    def train(self):
        for glob_iter in trange(self.num_glob_iters, desc="Global Rounds (pFedMe)"):
            # select participants
            self.selected_users = self.select_users(glob_iter, self.num_users)
            sel_ids = [u.id for u in self.selected_users]
            print(f"Exp {self.exp_no}: round {glob_iter} selected users: {sel_ids}")

            # clients: solve Moreau-regularized local problems -> personalized params
            personalized_param_lists = []
            weights = []
            for user in tqdm(self.selected_users, desc="Clients optimizing (Moreau)"):
                x_params = user.moreau_personalize(self.global_model, lamda=self.lamda)
                personalized_param_lists.append(x_params)
                weights.append(user.train_samples)

            # normalize weights
            wsum = sum(weights) if len(weights) > 0 else 1.0
            weights = [w / wsum for w in weights]

            # server meta update
            self.meta_update_from_personalized(personalized_param_lists, weights)

            # evaluate and save
            self.evaluate_local(glob_iter)
            self.evaluate_global(glob_iter)

        self.save_global_model(self.num_glob_iters - 1, tag="final")
        self.save_results()
