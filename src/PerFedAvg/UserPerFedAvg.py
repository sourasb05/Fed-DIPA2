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


class UserPerFedAvg:
    """
    Per-FedAvg client:
      - adapt_from_global(): inner-loop GD from the global init
      - adapt_then_eval_{val,test}(): re-adapt from global init and evaluate
      - local eval API preserved
    """
    def __init__(self, device, args, id, exp_no, current_directory, wandb_logger):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device
        self.wandb = wandb_logger

        self.id = id
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory

        # hyperparams
        self.inner_lr = args.alpha
        self.local_iters = args.local_iters
        self.num_glob_iters = args.num_global_iters
        self.algorithm = "PerFedAvg"
        self.country = args.country

        self.minimum_val_loss = float("inf")

        # data / features (copied from your template)
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

        # datasets / loaders
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

        # a throwaway model we can use for adaptation-eval without touching local_model
        self.eval_model = copy.deepcopy(self.local_model).to(self.device)

        # optimizer for local_model
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.inner_lr)

        # sample counts
        self.train_samples = train_size
        self.val_samples = val_size
        self.samples = train_size + val_size

        # metrics (same as your template)
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

        # per-round histories
        self.val_round_result_dict = {}
        self.val_global_round_result_dict = {}
        self.train_round_result_dict = {}
        self.train_global_round_result_dict = {}
        self.test_round_result_dict = {}
        self.test_global_round_result_dict = {}

    # ---------- parameter helpers ----------
    def set_parameters(self, src_model):
        for p, gp in zip(self.local_model.parameters(), src_model.parameters()):
            p.data = gp.data.clone()

    def load_into_eval_model(self, src_model):
        for ep, gp in zip(self.eval_model.parameters(), src_model.parameters()):
            ep.data = gp.data.clone()

    def get_parameters(self):
        return list(self.local_model.parameters())

    def get_parameter_list_copy(self):
        return [p.detach().clone() for p in self.local_model.parameters()]

    # ---------- inner-loop training ----------
    def _inner_loop(self, model, optimizer, steps, on_loader):
        model.train()
        for _ in range(steps):
            for batch in on_loader:
                features, addi, information, informativeness, sharingOwner, sharingOthers = batch
                optimizer.zero_grad()
                y_preds = model(features.to(self.device), addi.to(self.device))
                loss = model.compute_loss(y_preds, information, informativeness, sharingOwner, sharingOthers)
                loss.backward()
                optimizer.step()

    def adapt_from_global(self, global_model):
        """
        Start from global init -> run inner steps on train_loader -> return adapted params list.
        Also keep local_model at the adapted weights (personalization).
        """
        self.set_parameters(global_model)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.inner_lr)
        self._inner_loop(self.local_model, self.optimizer, self.local_iters, self.train_loader)
        return self.get_parameters()  # adapted params

    # ---------- evaluation that includes adaptation ----------
    @torch.no_grad()
    def _eval_loader(self, model, loader):
        results = []
        for vdata in loader:
            vdata = [x.to('cuda') for x in vdata]
            features, additional_info, information, informativeness, sharingOwner, sharingOthers = vdata
            y_preds = model(features, additional_info)
            results.append([information, informativeness, sharingOwner, sharingOthers, y_preds])
        return results

    def adapt_then_eval_val(self, global_model):
        """
        Copy global -> adapt on train -> evaluate on val
        """
        # work on eval_model to avoid touching personalized local_model
        self.load_into_eval_model(global_model)
        opt = torch.optim.Adam(self.eval_model.parameters(), lr=self.inner_lr)
        self._inner_loop(self.eval_model, opt, self.local_iters, self.train_loader)

        # now evaluate on val
        self.eval_model.eval()
        results = self._eval_loader(self.eval_model, self.val_loader)
        return self._metrics_from_results(results, dict_target="val_global_round_result_dict")

    def adapt_then_eval_test(self, global_model):
        """
        Copy global -> adapt on train -> evaluate on test
        """
        self.load_into_eval_model(global_model)
        opt = torch.optim.Adam(self.eval_model.parameters(), lr=self.inner_lr)
        self._inner_loop(self.eval_model, opt, self.local_iters, self.train_loader)

        self.eval_model.eval()
        results = self._eval_loader(self.eval_model, self.test_loader)
        return self._metrics_from_results(results, dict_target="test_global_round_result_dict")

    # ---------- local evals (no adaptation; use personalized local_model) ----------
    def test_local_model_val(self):
        self.local_model.eval()
        results = self._eval_loader(self.local_model, self.val_loader)
        return self._metrics_from_results(results, dict_target="val_round_result_dict")

    def test_local_model_test(self):
        self.local_model.eval()
        results = self._eval_loader(self.local_model, self.test_loader)
        return self._metrics_from_results(results, dict_target="test_round_result_dict")

    # ---------- metric helper ----------
    def _metrics_from_results(self, results, dict_target):
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'

        metrics = [Accuracy, Precision, Recall, F1Score]
        metrics_data = {
            m.__name__: [m(task="multilabel",
                           num_labels=od,
                           threshold=threshold,
                           average=average_method,
                           ignore_index=od - 1)
                         for _, od in output_channel.items()]
            for m in metrics
        }

        informativeness_scores = [[], []]  # gt, pred

        for information, informativeness, sharingOwner, sharingOthers, y_preds in results:
            gts = [information, sharingOwner, sharingOthers]
            output_dims = output_channel.values()
            for o, (output_dim, gt) in enumerate(zip(output_dims, gts)):
                start_dim = o * output_dim
                end_dim = start_dim + output_dim
                for metric_name in metrics_data.keys():
                    metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
            informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
            informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())

        results_data = {k: [i.compute().detach().cpu().numpy() for i in v]
                        for k, v in metrics_data.items()}
        result_dict = {key: [float(val) for val in value] for key, value in results_data.items()}

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(
            informativeness_scores[0], informativeness_scores[1]
        )
        # print(f"User ID: {self.id} {info_prec:.02f} {info_rec:.02f} {info_f1:.02f} {info_cmae:.02f} {info_mae:.02f}")

        # store per-round histories
        d = getattr(self, dict_target)
        if not d:
            setattr(self, dict_target, {k: [v] for k, v in result_dict.items()})
            d = getattr(self, dict_target)
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
