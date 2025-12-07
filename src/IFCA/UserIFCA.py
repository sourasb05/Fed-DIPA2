import os
import io
import math
import json
import copy
import torch
import shutil
import wandb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, CalibrationError

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from dipa2.ML_baseline.Study2TransMLP.inference_dataset import ImageMaskDataset
from src.TrainModels.trainmodels import PrivacyModel
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics


class UserIFCA:
    """
    IFCA client. Can:
      - score K cluster models and pick best (E-step),
      - train on chosen cluster model (M-step),
      - evaluate local and global (cluster) models (reuse your metric flows).
    """
    def __init__(self, device, args, id, exp_no, current_directory, wandb_logger):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb_logger

        # identity & paths
        self.id = id
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory

        # hparams
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters
        self.num_glob_iters = args.num_global_iters
        self.algorithm = "IFCA"
        self.country = args.country

        self.minimum_val_loss = float("inf")
        self.distance = 0.0

        # personalization / data prep (copied from your template)
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

        # Split
        num_rows = len(self.mega_table)
        if num_rows <= 3:
            self.valid = False
            return
        self.valid = True

        train_per, val_per, test_per = 65, 10, 25
        train_size = math.floor((train_per / 100.0) * num_rows)
        val_size = math.ceil((val_per / 100.0) * num_rows)
        test_size = num_rows - train_size - val_size

        train_df = self.mega_table.sample(n=train_size, random_state=0)
        rem_df = self.mega_table.drop(train_df.index)
        val_df = rem_df.sample(n=val_size, random_state=0)
        test_df = rem_df.drop(val_df.index)

        dataset_files_dir = f"dataset_files/{self.algorithm}/"
        os.makedirs(dataset_files_dir, exist_ok=True)
        train_df.to_csv(f"{dataset_files_dir}/train_{int(self.id)}_.csv", index=False)
        val_df.to_csv(f"{dataset_files_dir}/val_{int(self.id)}_.csv", index=False)
        test_df.to_csv(f"{dataset_files_dir}/test_{int(self.id)}_.csv", index=False)

        # Load datasets/loaders
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

        # Models
        self.local_model = PrivacyModel(
            input_dim=self.input_dim,
            max_bboxes=test_dataset.max_bboxes,
            features_dim=test_dataset.features_dim,
        ).to(self.device)

        # a separate evaluation copy we can load cluster weights into without touching local_model
        self.eval_model = copy.deepcopy(self.local_model).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        # Sample counts
        self.train_samples = train_size
        self.val_samples = val_size
        self.samples = train_size + val_size

        # Metrics (copied from your template)
        threshold = 0.5
        average_method = 'micro'
        self.acc = [Accuracy(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.pre = [Precision(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.rec = [Recall(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                    for _, od in self.output_channel.items()]
        self.f1 = [F1Score(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                   for _, od in self.output_channel.items()]
        self.conf = [ConfusionMatrix(task="multilabel", num_labels=od)
                     for _, od in self.output_channel.items()]

        self.global_acc = [Accuracy(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_pre = [Precision(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_rec = [Recall(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                           for _, od in self.output_channel.items()]
        self.global_f1 = [F1Score(task="multilabel", num_labels=od, threshold=threshold, average=average_method, ignore_index=od - 1)
                          for _, od in self.output_channel.items()]
        self.global_conf = [ConfusionMatrix(task="multilabel", num_labels=od)
                            for _, od in self.output_channel.items()]

        # per-round result dicts (kept)
        self.val_round_result_dict = {}
        self.val_global_round_result_dict = {}
        self.train_round_result_dict = {}
        self.train_global_round_result_dict = {}
        self.test_round_result_dict = {}
        self.test_global_round_result_dict = {}

        # current cluster assignment
        self.assigned_cluster = None

    # ---------- Parameter helpers ----------
    def set_parameters(self, global_model):
        for param, glob_param in zip(self.local_model.parameters(), global_model.parameters()):
            param.data = glob_param.data.clone()

    def load_into_eval_model(self, global_model):
        for eparam, gparam in zip(self.eval_model.parameters(), global_model.parameters()):
            eparam.data = gparam.data.clone()

    def get_parameters(self):
        return self.local_model.parameters()

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()

    # ---------- IFCA E-step ----------
    @torch.no_grad()
    def _avg_train_loss_on_model(self, model):
        """
        Evaluate average training loss on a given model (no grad).
        We use full-train loader (stable). You can switch to val_loader if preferred.
        """
        self.eval_model.eval()
        self.load_into_eval_model(model)
        total_loss = 0.0
        total_items = 0
        for vdata in self.trainloaderfull:
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.eval_model(features.to(self.device), additional_information.to(self.device))
            loss = self.eval_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item() * len(features)
            total_items += len(features)
        return total_loss / max(1, total_items)

    def select_cluster(self, cluster_models):
        """
        Score all cluster models and select the best (lowest loss) for this user.
        Returns (k_star, list_of_losses)
        """
        losses = []
        for model in cluster_models:
            loss = self._avg_train_loss_on_model(model)
            losses.append(loss)
        k_star = int(np.argmin(losses))
        self.assigned_cluster = k_star
        return k_star, losses

    # ---------- M-step local training ----------
    def evaluate_model_and_maybe_save(self, t, epoch):
        self.local_model.eval()
        total_loss = 0.0

        for vdata in self.val_loader:
            features, additional_information, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
            loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        # Save per-user local best checkpoint
        self._save_local_model_if_better(t, avg_loss)

    def _save_local_model_if_better(self, glob_iter, current_loss):
        model_path = os.path.join(
            self.current_directory, "models", self.algorithm, "local_model",
            f"_GE_{self.num_glob_iters}_LE_{self.local_iters}", f"_user_{self.id}"
        )
        os.makedirs(model_path, exist_ok=True)

        if current_loss < self.minimum_val_loss:
            self.minimum_val_loss = current_loss
            checkpoint = {'GR': glob_iter,
                          'model_state_dict': self.local_model.state_dict(),
                          'loss': self.minimum_val_loss}
            torch.save(checkpoint, os.path.join(model_path, "best_local_checkpoint.pt"))

        if glob_iter == self.num_glob_iters - 1:
            checkpoint = {'GR': glob_iter,
                          'model_state_dict': self.local_model.state_dict(),
                          'loss': self.minimum_val_loss}
            torch.save(checkpoint, os.path.join(model_path, f"local_checkpoint_GR{glob_iter}.pt"))

    def train_on_cluster(self, t, cluster_model):
        """
        M-step: start from cluster model, run local SGD (no proximal term).
        """
        # initialize local params from the cluster model
        self.set_parameters(cluster_model)

        self.local_model.train()
        for _ in range(self.local_iters):
            for batch in self.train_loader:
                features, additional_information, information, informativeness, sharingOwner, sharingOthers = batch
                self.optimizer.zero_grad()
                y_preds = self.local_model(features.to(self.device), additional_information.to(self.device))
                loss = self.local_model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss.backward()
                self.optimizer.step()

        # optional per-epoch eval + save best by val loss
        self.evaluate_model_and_maybe_save(t, epoch=self.local_iters - 1)

    # ---------- The rest (evaluation) stays identical to your client ----------
    def test_eval(self):
        self.local_model.eval()
        results = []
        for vdata in self.val_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return results

    def _ensure_dict(self, name: str):
        """Ensure self.<name> exists and is a dict; return it."""
        d = getattr(self, name, None)
        if not isinstance(d, dict):
            setattr(self, name, {})
            d = getattr(self, name)
        return d

    def test_global_model_val(self, global_model):
        # --- use eval_model so we don't clobber local personalization ---
        self.load_into_eval_model(global_model)
        self.eval_model.eval()

        results = []
        for vdata in self.val_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.eval_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        # === compute metrics (unchanged) ===
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        from torchmetrics import Accuracy, Precision, Recall, F1Score
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {m.__name__: [m(task="multilabel", num_labels=od, threshold=threshold,
                                    average=average_method, ignore_index=od-1)
                                    for _, od in output_channel.items()]
                        for m in metrics}
        informativeness_scores = [[], []]

        for information, informativeness, sharingOwner, sharingOthers, y_preds in results:
            gts = [information, sharingOwner, sharingOthers]
            for o, (output_dim, gt) in enumerate(output_channel.items() if False else output_channel.items()):
                pass  # placeholder to keep structure; we use the loop below
            output_dims = list(output_channel.values())
            for o, (output_dim, gt) in enumerate(zip(output_dims, [information, sharingOwner, sharingOthers])):
                s = o * output_dim
                e = s + output_dim
                for mname in metrics_data.keys():
                    metrics_data[mname][o].update(y_preds[:, s:e], gt)

            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())

        results_data = {k: [i.compute().detach().cpu().numpy() for i in v]
                        for k, v in metrics_data.items()}
        result_dict = {key: [float(val) for val in value] for key, value in results_data.items()}

        from src.utils.results_utils import InformativenessMetrics
        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(
            informativeness_scores[0], informativeness_scores[1]
        )
        # (f"User ID: {self.id} {info_prec:.02f} {info_rec:.02f} {info_f1:.02f} {info_cmae:.02f} {info_mae:.02f}")

        # === robust init/append into the right container ===
        d = self._ensure_dict("val_global_round_result_dict")
        if not d:  # first time: store each metric as a list-of-lists
            for k, v in result_dict.items():
                d[k] = [v]          # wrap in a list so future rounds can append
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


    def test_global_model_test(self, global_model):
        # --- use eval_model so we don't clobber local personalization ---
        self.load_into_eval_model(global_model)
        self.eval_model.eval()

        results = []
        for vdata in self.test_loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.eval_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        # === compute metrics (unchanged) ===
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        from torchmetrics import Accuracy, Precision, Recall, F1Score
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {m.__name__: [m(task="multilabel", num_labels=od, threshold=threshold,
                                    average=average_method, ignore_index=od-1)
                                    for _, od in output_channel.items()]
                        for m in metrics}
        informativeness_scores = [[], []]

        output_dims = list(output_channel.values())
        for information, informativeness, sharingOwner, sharingOthers, y_preds in results:
            for o, (output_dim, gt) in enumerate(zip(output_dims, [information, sharingOwner, sharingOthers])):
                s = o * output_dim
                e = s + output_dim
                for mname in metrics_data.keys():
                    metrics_data[mname][o].update(y_preds[:, s:e], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())

        results_data = {k: [i.compute().detach().cpu().numpy() for i in v]
                        for k, v in metrics_data.items()}
        result_dict = {key: [float(val) for val in value] for key, value in results_data.items()}

        from src.utils.results_utils import InformativenessMetrics
        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(
            informativeness_scores[0], informativeness_scores[1]
        )
        # print(f"User ID: {self.id} {info_prec:.02f} {info_rec:.02f} {info_f1:.02f} {info_cmae:.02f} {info_mae:.02f}")

        # === robust init/append into the right container ===
        d = self._ensure_dict("test_global_round_result_dict")
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

    # ---------- Local-model evals (unchanged) ----------
    def test_local_model_test(self):
        # identical to your template; omitted here for brevity (same body)
        return self._test_local_common(self.test_loader, dict_name="test_round_result_dict")

    def test_local_model_val(self):
        return self._test_local_common(self.val_loader, dict_name="val_round_result_dict")

    def _test_local_common(self, loader, dict_name):
        self.local_model.eval()
        results = []
        for vdata in loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            with torch.no_grad():
                y_preds = self.local_model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])

        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=od,
                                                    threshold=threshold,
                                                    average=average_method,
                                                    ignore_index=od - 1)
                                            for _, od in output_channel.items()]
        informativeness_scores = [[], []]

        for result in results:
            information, informativeness, sharingOwner, sharingOthers, y_preds = result
            gts = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gts)):
                start_dim = o * (output_dim)
                end_dim = o * (output_dim) + output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())

        results_data = {k: [i.compute().detach().cpu().numpy() for i in v] for k, v in metrics_data.items()}
        result_dict = {key: [float(val) for val in value] for key, value in results_data.items()}

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(
            informativeness_scores[0], informativeness_scores[1]
        )

        # init/append
        d = getattr(self, dict_name)
        if not d:
            setattr(self, dict_name, {k: [v] for k, v in result_dict.items()})
            d = getattr(self, dict_name)
            d.update({'info_prec': [info_prec],
                    'info_rec': [info_rec],
                    'info_f1': [info_f1],
                    'info_cmae': [info_cmae],
                    'info_mae': [info_mae]})
        else:
            for k in result_dict:
                d[k].append(result_dict[k])
            d['info_prec'].append(info_prec)
            d['info_rec'].append(info_rec)
            d['info_f1'].append(info_f1)
            d['info_cmae'].append(info_cmae)
            d['info_mae'].append(info_mae)

        return info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict
