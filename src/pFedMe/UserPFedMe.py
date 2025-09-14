import os
import math
import json
import copy
import wandb
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from src.utils.results_utils import InformativenessMetrics


class UserPFedMe:
    """
    pFedMe client:
      - keeps a persistent personalized model x_i
      - each round: minimize F_i(x) + (lambda/2)||x - w||^2 for local_iters steps
      - returns x_i params to the server for the meta update; keeps x_i for local eval
    """

    def __init__(self, device, args, id, exp_no, current_directory, wandb_logger):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb_logger

        self.id = id
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory

        # Hyperparameters
        self.inner_lr = args.alpha
        self.local_iters = args.local_iters
        self.num_glob_iters = args.num_global_iters
        self.algorithm = "pFedMe"
        self.country = args.country
        self.lamda = args.lamda_sim_sta

        self.minimum_val_loss = float("inf")

        # Data / features (copied from your template)
        self.bigfives = ["extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness"]
        self.basic_info = ["age", "gender", "nationality", "frequency"]
        self.category = ["category"]
        self.privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']

        self.mega_table = pd.read_csv(current_directory + f'/feature_clients/annotations_annotator{self.id}.csv')
        self.encoder = LabelEncoder()
        for col in ['category', 'gender', 'platform', 'originalDataset', 'nationality']:
            self.mega_table[col] = self.encoder.fit_transform(self.mega_table[col])

        self.input_channel = []
        self.input_channel.extend(self.basic_info)
        self.input_channel.extend(self.category)
        self.input_channel.extend(self.bigfives)
        self.input_dim = len(self.input_channel)

        self.output_name = self.privacy_metrics
        self.output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}

        image_size = (224, 224)

        # split
        num_rows = len(self.mega_table)
        if num_rows <= 3:
            self.valid = False
            return
        self.valid = True

        train_per, val_per, test_per = 65, 10, 25
        train_size = math.floor((train_per/100.0) * num_rows)
        val_size = math.ceil((val_per/100.0) * num_rows)
        test_size = num_rows - train_size - val_size

        train_df = self.mega_table.sample(n=train_size, random_state=0)
        rem_df = self.mega_table.drop(train_df.index)
        val_df = rem_df.sample(n=val_size, random_state=0)
        test_df = rem_df.drop(val_df.index)

        dataset_files_dir = f"dataset_files/{self.algorithm}/"
        os.makedirs(dataset_files_dir, exist_ok=True)
        train_df.to_csv(f"{dataset_files_dir}/train_{int(self.id)}.csv", index=False)
        val_df.to_csv(f"{dataset_files_dir}/val_{int(self.id)}.csv", index=False)
        test_df.to_csv(f"{dataset_files_dir}/test_{int(self.id)}.csv", index=False)

        # datasets/loaders
        train_dataset = ImageMaskDataset(train_df, args.model_name, self.input_channel, image_size, flip=True)
        val_dataset = ImageMaskDataset(val_df, args.model_name, self.input_channel, image_size)
        test_dataset = ImageMaskDataset(test_df, args.model_name, self.input_channel, image_size)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       generator=torch.Generator(device='cuda'), shuffle=True)
        self.trainloaderfull = DataLoader(train_dataset, batch_size=len(train_dataset),
                                          generator=torch.Generator(device='cuda'), shuffle=False)
        self.val_loader = DataLoader(val_dataset, generator=torch.Generator(device='cuda'),
                                     batch_size=len(val_dataset))
        self.test_loader = DataLoader(test_dataset, generator=torch.Generator(device='cuda'),
                                      batch_size=16)

        # models
        self.local_model = PrivacyModel(
            input_dim=self.input_dim,
            max_bboxes=test_dataset.max_bboxes,
            features_dim=test_dataset.features_dim,
        ).to(self.device)

        # separate eval copy so we don't clobber personalization during "global" eval
        self.eval_model = copy.deepcopy(self.local_model).to(self.device)

        # optimizer for local_model
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.inner_lr)

        # counts
        self.train_samples = train_size
        self.val_samples = val_size
        self.samples = train_size + val_size

        # metrics (as in your template)
        threshold = 0.5
        average_method = 'micro'
        self.acc = [Accuracy(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.pre = [Precision(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.rec = [Recall(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.f1  = [F1Score(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.conf = [ConfusionMatrix(task="multilabel", num_labels=od)
                     for _, od in self.output_channel.items()]

        self.global_acc = [Accuracy(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_pre = [Precision(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_rec = [Recall(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_f1  = [F1Score(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_conf = [ConfusionMatrix(task="multilabel", num_labels=od)
                            for _, od in self.output_channel.items()]

        # histories
        self.val_round_result_dict = {}
        self.val_global_round_result_dict = {}
        self.train_round_result_dict = {}
        self.train_global_round_result_dict = {}
        self.test_round_result_dict = {}
        self.test_global_round_result_dict = {}

        self._first_round_init = True  # to warm-start from global on the very first round only

    # ---------- parameter helpers ----------
    def set_parameters(self, src_model):
        for p, gp in zip(self.local_model.parameters(), src_model.parameters()):
            p.data = gp.data.clone()

    def load_into_eval_model(self, src_model):
        for ep, gp in zip(self.eval_model.parameters(), src_model.parameters()):
            ep.data = gp.data.clone()

    def get_parameter_list_copy(self):
        return [p.detach().clone() for p in self.local_model.parameters()]

    # ---------- local Moreau-regularized personalization ----------
    def moreau_personalize(self, global_model, lamda: float):
        """
        Minimize F_i(x) + (lamda/2)||x - w||^2 for self.local_iters steps starting
        from current personalized weights (first round starts from global).
        Returns a list of Tensors: the personalized params x_i after optimization.
        """
        if self._first_round_init:
            self.set_parameters(global_model)
            self._first_round_init = False

        self.local_model.train()
        # Re-instantiate optimizer to ensure lr is inner_lr every round
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.inner_lr)

        for _ in range(self.local_iters):
            for batch in self.train_loader:
                features, addi, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(features.to(self.device), addi.to(self.device))
                loss_task = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)

                # Moreau / Prox to current global w
                prox = 0.0
                for p_local, p_global in zip(self.local_model.parameters(), global_model.parameters()):
                    prox += 0.5 * lamda * torch.norm(p_local - p_global)**2

                loss = loss_task + prox
                loss.backward()
                self.optimizer.step()

        # Optionally evaluate & save best by val loss
        self.evaluate_model_and_maybe_save()

        # Return detached personalized parameters
        return self.get_parameter_list_copy()

    # ---------- eval helpers ----------
    def evaluate_model_and_maybe_save(self):
        self.local_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for vdata in self.val_loader:
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
                y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                total_loss += loss.item()
        avg_loss = total_loss / max(1, len(self.val_loader))
        if avg_loss < self.minimum_val_loss:
            self.minimum_val_loss = avg_loss
            model_path = os.path.join(
                self.current_directory, "models", self.algorithm, "local_model",
                f"_GE_{self.num_glob_iters}_LE_{self.local_iters}", f"_user_{self.id}"
            )
            os.makedirs(model_path, exist_ok=True)
            checkpoint = {'model_state_dict': self.local_model.state_dict(), 'loss': self.minimum_val_loss}
            torch.save(checkpoint, os.path.join(model_path, "best_local_checkpoint.pt"))

    # ---------- "global" evals (copy w into eval_model; do NOT touch personalization) ----------
    def _ensure_dict(self, name: str):
        d = getattr(self, name, None)
        if not isinstance(d, dict):
            setattr(self, name, {})
            d = getattr(self, name)
        return d

    def test_global_model_val(self, global_model):
        self.load_into_eval_model(global_model)
        self.eval_model.eval()

        results = []
        for vdata in self.val_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.eval_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        return self._metrics_to_dict(results, dict_name="val_global_round_result_dict")

    def test_global_model_test(self, global_model):
        self.load_into_eval_model(global_model)
        self.eval_model.eval()

        results = []
        for vdata in self.test_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.eval_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        return self._metrics_to_dict(results, dict_name="test_global_round_result_dict")

    # ---------- personalized local evals ----------
    def test_local_model_val(self):
        self.local_model.eval()
        results = []
        for vdata in self.val_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return self._metrics_to_dict(results, dict_name="val_round_result_dict")

    def test_local_model_test(self):
        self.local_model.eval()
        results = []
        for vdata in self.test_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return self._metrics_to_dict(results, dict_name="test_round_result_dict")

    # ---------- metric helper ----------
    def _metrics_to_dict(self, results, dict_name):
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {m.__name__: [m(task="multilabel",
                                       num_labels=od,
                                       threshold=threshold,
                                       average=average_method,
                                       ignore_index=od - 1)
                                     for _, od in output_channel.items()]
                        for m in metrics}
        informativeness_scores = [[], []]

        output_dims = list(output_channel.values())
        for information, informativeness, sharingOwner, sharingOthers, y_preds in results:
            for o, (od, gt) in enumerate(zip(output_dims, [information, sharingOwner, sharingOthers])):
                s, e = o * od, o * od + od
                for mname in metrics_data.keys():
                    metrics_data[mname][o].update(y_preds[:, s:e], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())

        results_data = {k: [i.compute().detach().cpu().numpy() for i in v] for k, v in metrics_data.items()}
        result_dict = {key: [float(val) for val in value] for key, value in results_data.items()}

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(
            informativeness_scores[0], informativeness_scores[1]
        )
        # print(f"User ID: {self.id} {info_prec:.02f} {info_rec:.02f} {info_f1:.02f} {info_cmae:.02f} {info_mae:.02f}")

        d = self._ensure_dict(dict_name)
        if not d:
            for k, v in result_dict.items():
                d[k] = [v]
            d['info_prec'] = [float(info_prec)]
            d['info_rec']  = [float(info_rec)]
            d['info_f1']   = [float(info_f1)]
            d['info_cmae'] = [float(info_cmae)]
            d['info_mae']  = [float(info_mae)]
        else:
            for k, v in result_dict.items():
                d[k].append(v)
            d['info_prec'].append(float(info_prec))
            d['info_rec'].append(float(info_rec))
            d['info_f1'].append(float(info_f1))
            d['info_cmae'].append(float(info_cmae))
            d['info_mae'].append(float(info_mae))

        return info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict
