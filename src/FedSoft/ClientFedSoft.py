import os
import io
import math
import json
import copy
import torch
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics


class UserFedSoft:
    """
    FedSoft client:
      - Builds the same dataset/loaders as your FedProx client.
      - Maintains u_{k,s} (soft importance weights) and recomputes them every τ rounds on server request.
      - local_prox_update(): solves h_k(w) = f_k(w) + (λ/2) * sum_s u_{k,s} || w - c_s ||^2  for local_iters epochs.
    """
    def __init__(self, device, args, id, exp_no, current_directory, wandb,
                 num_clusters: int, lamda_prox: float, sigma: float):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb

        self.id = id
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory

        # Hyperparams
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters
        self.num_glob_iters = args.num_global_iters
        self.eta = args.eta
        self.fixed_user_id = args.fixed_user_id
        self.country = args.country

        # FedSoft-specific
        self.S = num_clusters
        self.lamda = lamda_prox
        self.sigma = sigma

        self.algorithm = "FedSoft"
        self.minimum_val_loss = float('inf')

        # Columns & encoders (same as your FedProx client)
        self.bigfives = ["extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness"]
        self.basic_info = ["age", "gender", "nationality", "frequency"]
        self.category = ["category"]
        self.privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']
        self.mega_table = pd.read_csv(current_directory + f'/feature_clients/annotations_annotator{self.id}.csv')

        self.description = {'informationType': ['It tells personal information',
                                                'It tells location of shooting',
                                                'It tells individual preferences/pastimes',
                                                'It tells social circle',
                                                'It tells others\' private/confidential information',
                                                'Other things'],
                            'informativeness': ['Strongly disagree', 'Disagree', 'Slightly disagree', 'Neither',
                                                'Slightly agree', 'Agree', 'Strongly agree'],
                            'sharingOwner': ['I won\'t share it', 'Close relationship', 'Regular relationship',
                                             'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
                            'sharingOthers': ['I won\'t allow others to share it', 'Close relationship', 'Regular relationship',
                                              'Acquaintances', 'Public', 'Broadcast program', 'Other recipients']}

        self.encoder = LabelEncoder()
        self.mega_table['category'] = self.encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = self.encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = self.encoder.fit_transform(self.mega_table['platform'])
        self.mega_table['originalDataset'] = self.encoder.fit_transform(self.mega_table['originalDataset'])
        self.mega_table['nationality'] = self.encoder.fit_transform(self.mega_table['nationality'])

        self.input_channel = []
        self.input_channel.extend(self.basic_info)
        self.input_channel.extend(self.category)
        self.input_channel.extend(self.bigfives)
        self.input_dim = len(self.input_channel)

        self.output_name = self.privacy_metrics
        self.output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}

        # Dataset splits
        num_rows = len(self.mega_table)
        if num_rows <= 3:
            self.valid = False
            return
        self.valid = True

        train_per, val_per, test_per = 65, 10, 25
        train_size = math.floor((train_per/100.0) * num_rows)
        val_size   = math.ceil((val_per/100.0) * num_rows)
        test_size  = num_rows - train_size - val_size

        train_df = self.mega_table.sample(n=train_size, random_state=0)
        rem_df   = self.mega_table.drop(train_df.index)
        val_df   = rem_df.sample(n=val_size, random_state=0)
        test_df  = rem_df.drop(val_df.index)

        dataset_files_dir = f"dataset_files/{self.algorithm}/"
        os.makedirs(dataset_files_dir, exist_ok=True)
        train_df.to_csv(f"{dataset_files_dir}/train_{int(self.id)}.csv", index=False)
        val_df.to_csv(f"{dataset_files_dir}/val_{int(self.id)}.csv", index=False)
        test_df.to_csv(f"{dataset_files_dir}/test_{int(self.id)}.csv", index=False)

        image_size = (224, 224)
        if not args.test:
            train_dataset = ImageMaskDataset(train_df, args.model_name, self.input_channel, image_size, flip=True)
            val_dataset   = ImageMaskDataset(val_df,   args.model_name, self.input_channel, image_size)
            self.train_loader      = DataLoader(train_dataset, batch_size=self.batch_size,
                                                generator=torch.Generator(device='cuda'), shuffle=True)
            self.trainloaderfull   = DataLoader(train_dataset, batch_size=len(train_dataset),
                                                generator=torch.Generator(device='cuda'), shuffle=False)
            self.val_loader        = DataLoader(val_dataset, generator=torch.Generator(device='cuda'),
                                                batch_size=len(val_dataset))
        test_dataset  = ImageMaskDataset(test_df,  args.model_name, self.input_channel, image_size)
        self.test_loader = DataLoader(test_dataset, generator=torch.Generator(device='cuda'), batch_size=16)

        # Local & eval models
        self.local_model = PrivacyModel(
            input_dim=self.input_dim,
            max_bboxes=test_dataset.max_bboxes,
            features_dim=test_dataset.features_dim
        ).to(self.device)

        self.eval_model = copy.deepcopy(self.local_model)
        self.optimizer  = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        self.train_samples = train_size
        self.val_samples   = val_size
        self.samples       = train_size + val_size

        # Soft-weights u_{k,s} (initialized uniform small, then estimated)
        self.u_ks = [1.0 / self.S] * self.S
        self.n_k  = self.train_samples  # used for weighting in server aggregation

        # Slot for centers
        self.centers = None  # list of models length S

        # Metric objects (kept from your client for later reuse if desired)
        threshold = 0.5
        average_method = 'micro'
        self.global_acc = [Accuracy(task="multilabel", num_labels=od, threshold=threshold,
                                    average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_pre = [Precision(task="multilabel", num_labels=od, threshold=threshold,
                                     average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_rec = [Recall(task="multilabel", num_labels=od, threshold=threshold,
                                  average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_f1  = [F1Score(task="multilabel", num_labels=od, threshold=threshold,
                                   average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]

        # Round result dicts (optional; not heavily used here to keep code focused)
        self.val_round_result_dict = {}
        self.val_global_round_result_dict = {}
        self.test_round_result_dict = {}
        self.test_global_round_result_dict = {}

        if args.test:
            self.load_model()

    # -----------------------
    # Wiring helpers
    # -----------------------
    def load_model(self):
        models_dir = "./models/FedAvg/global_model/"
        model_state_dict = torch.load(os.path.join(models_dir, "server_checkpoint.pt"))["model_state_dict"]
        self.local_model.load_state_dict(model_state_dict)
        self.local_model.eval()

    def set_centers(self, centers):
        """Receive centers from server (deepcopy for safety)."""
        self.centers = [copy.deepcopy(c).to(self.device) for c in centers]

    # -----------------------
    # Importance-weight estimation (Algorithm 1, lines 3–14)
    # For each local sample, find s = argmin_s loss(c_s; x, y). Count per-s.
    # u_{k,s} = max( n_{k,s} / n_k , σ )
    # -----------------------
    @torch.no_grad()
    def estimate_importance_weights(self):
        assert self.centers is not None, "Centers not set before importance-weight estimation."

        counts = np.zeros(self.S, dtype=np.int64)
        total = 0

        # Use validation (or train) loader for classification of samples to centers.
        # Here we use val_loader; if empty, fall back to trainloaderfull.
        loader = self.val_loader if len(getattr(self, "val_loader", [])) > 0 else self.trainloaderfull

        for batch in loader:
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = batch
            features = features.to(self.device)
            additional_info = additional_info.to(self.device)

            # Compute per-center loss for this batch and take argmin per-sample
            # We'll approximate per-sample argmin by comparing total batch loss for each center
            # and "assign" the whole batch to the lowest-loss center (lightweight; can be made per-sample if needed).
            losses_per_center = []
            for s in range(self.S):
                y_preds = self.centers[s](features, additional_info)
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                losses_per_center.append(loss.item())

            j = int(np.argmin(losses_per_center))
            bsz = features.size(0)
            counts[j] += bsz
            total += bsz

        # Smooth + normalize
        if total == 0:
            # fallback uniform with smoothing
            self.u_ks = [max(1.0 / self.S, self.sigma) for _ in range(self.S)]
            self.n_k = self.train_samples
            return

        raw = counts.astype(float) / float(total)
        self.u_ks = [max(float(raw[s]), float(self.sigma)) for s in range(self.S)]
        self.n_k = self.train_samples  # used by server in weighting

    # -----------------------
    # ONE local solve of the proximal objective:
    #   h_k(w) = f_k(w) + (λ/2) * Σ_s u_{k,s} || w - c_s ||^2
    # We follow your train loop and simply add ALL center penalties each step.
    # -----------------------
    def local_prox_update(self):
        assert self.centers is not None, "Centers not set before local_prox_update."

        self.local_model.train()
        opt = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        for _ in range(self.local_iters):
            for batch in self.train_loader:
                features, additional_info, information, informativeness, sharingOwner, sharingOthers = batch
                features = features.to(self.device)
                additional_info = additional_info.to(self.device)

                opt.zero_grad()
                y_preds = self.local_model(features, additional_info)
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)

                # Proximal term across all centers with soft weights u_{k,s}
                prox_term = 0.0
                with torch.no_grad():
                    # collect center params into tensors for subtraction
                    center_params = [[p.detach() for p in c.parameters()] for c in self.centers]
                for p_idx, p in enumerate(self.local_model.parameters()):
                    # Sum_s u_{k,s} ||p - c_s[p_idx]||^2
                    diff_sq_sum = 0.0
                    for s in range(self.S):
                        diff = p - center_params[s][p_idx]
                        diff_sq_sum = diff_sq_sum + self.u_ks[s] * torch.sum(diff * diff)
                    prox_term = prox_term + 0.5 * self.lamda * diff_sq_sum

                total_loss = loss + prox_term
                total_loss.backward()
                opt.step()

        # Return a deepcopy of the updated local model for server aggregation
        return copy.deepcopy(self.local_model).to(self.device)
