import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

from config import TFPPConfig
from dataloader import TFPPDataset, get_dataloader
from model import TFPP
from trainer._scheduler import WarmupSequentialLR
from utils import (
    Logger,
    MSEOffsetLoss,
    MSEPCCLoss,
    calculate_metrics_by_average,
    calculate_phenotype_metrics_by_time,
    denormalize_phenotype,
    load_scalers,
    save_model_config,
    save_scalers,
)
from utils.Genotype import (
    GENOTYPE_ENCODING_VERSION,
    TrainingDosageImputer,
    model_codes_to_dosage,
    prepare_screening_dosage,
)
from utils.Screening import (
    SCREENING_ALGORITHM_VERSION,
    array_fingerprint,
    training_indices_hash,
)
from utils.Split import (
    IndividualSplit, default_split_path, load_or_create_split, load_split, save_split,
    split_from_indices,
)


class TFPPTrainer:
    """表型预测 Transformer 训练器。

    基于 Transformer 架构的 SNP 表型预测模型训练器，支持常规训练与 K 折交叉验证两种模式。

    Parameters
    ----------
    config : TFPPConfig
        训练配置对象，包含模型结构、训练超参数、数据路径等全部设置。

    Notes
    -----
    主要功能:

    - 训练基于 Transformer 的 SNP 表型预测模型
    - 使用多种评估指标 (PCC, R², MSE, RMSE) 全面评估模型性能
    - 自动保存最佳模型和训练统计信息
    - 支持早停和学习率调度
    """

    def __init__(self, config: TFPPConfig):
        """初始化训练器并构建数据集与模型。"""
        # 保存配置
        self.config = config
        existing = Path(config.output_dir)
        if (existing / "models/best_model.pt").exists() or (existing / "test_evaluation.json").exists():
            raise FileExistsError("实验已有 checkpoint/test 预约，禁止覆盖或再次观察 test；请使用新实验目录或 runner --resume")

        # 保存目录参数
        self.pheno_dir = self.config.pheno_dir
        self.snp_file = self.config.snp_file

        self.best_model_save_dir = Path(config.best_model_save_dir)
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)

        self.training_history_save_dir = Path(config.training_history_save_dir)
        self.training_history_save_dir.mkdir(parents=True, exist_ok=True)

        # 实验参数
        self.exp_name = config.exp_name

        # 路径参数
        self.scaler_save_dir = Path(config.scaler_save_dir)
        self.scaler_save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_file = str(self.scaler_save_dir / "scaler.pkl")

        # 数据集初始化
        self.phenotype_names = config.phenotype_names
        self.dataset = self._create_dataset()
        self.num_sample = len(self.dataset)
        if self.dataset.phenotype.shape[2] != len(config.time_coordinates):
            raise ValueError("phenotype 时间列数必须与 time_coordinates 一致")
        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio
        self.norm_method = config.norm_method
        self.scaler_type = config.scaler_type
        self.seed = config.seed

        # 日志记录器（在筛选前创建，使筛选日志也写入同一文件）
        self.log_file = self.config.log_file
        self.logger = Logger(self.log_file, unique=False)

        # 普通训练和每个折实验均先划分再筛选；外层协调器不筛选或初始化模型。
        self.screen_k = config.screen_k
        self.gwas_k = config.gwas_k
        self.screen_multivariate = config.screen_multivariate
        self.screen_missing_strategy = config.screen_missing_strategy
        self.logger.info(
            f"筛选配置: screen_multivariate={self.screen_multivariate}, "
            f"screen_missing_strategy={self.screen_missing_strategy}"
        )
        self.use_kfold = config.use_kfold
        self.n_folds = config.n_folds
        self.original_snp_dim = self.dataset.snp.shape[1]
        self.wide_fill_values = None

        # 训练超参数
        self.epochs = config.epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        # 梯度累积步数: 将等效批量固定为 32, 即 accumulation_steps = 32 / batch_size
        # max(1, ...) 防止 batch_size > 32 时除零
        self.accumulation_steps = max(1, 32 // self.batch_size)
        self.patience = config.patience
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.num_workers = config.num_workers
        self.dropout = config.dropout

        # 学习率调度器参数
        self.use_warmup = config.use_warmup
        self.warmup_epochs = config.warmup_epochs if self.use_warmup else 0
        self.warmup_start_factor = (
            config.warmup_start_factor if self.use_warmup else 0.1
        )
        self.warmup_end_factor = config.warmup_end_factor if self.use_warmup else 1.0
        self.min_lr = config.min_lr

        # 是否保存训练信息
        self.is_save_training_info = config.is_save_training_info

        # 损失函数
        self.loss_type = config.loss_type
        self.offset_weight = config.offset_weight
        self.pcc_weight = config.pcc_weight
        self.l1_wide_weight = config.l1_wide_weight

        # 模型配置
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.num_embeddings = config.num_embeddings
        self.d_ff = config.d_ff
        self.degree = config.degree
        self.ablation_mode = config.ablation_mode
        self.time_coordinates = config.time_coordinates

        # 外层协调器保留原始矩阵；所有拟合及模型创建由独立折实验执行。
        if self.use_kfold:
            if self.n_folds > self.num_sample:
                raise ValueError("n_folds 不能超过个体数")
            self.snp_dim = self.original_snp_dim
            self.model = None
        else:
            self._prepare_regular_data()
            torch.manual_seed(config.seed)
            self.model = self._create_model()

    def _prepare_regular_data(self):
        """所有依赖标签的拟合只接收固定 train_idx，模型输入始终保留缺失类别0。"""
        from utils import compute_cache_key, screen_snps

        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        split_path = self.config.split_file
        if split_path is None or split_path == "auto":
            split_path = default_split_path(
                self.num_sample, ratios=ratios, seed=self.seed
            )
        self.split = load_or_create_split(
            split_path, self.num_sample, ratios=ratios, seed=self.seed
        )
        train_idx = self.split.train_idx
        if min(len(train_idx), len(self.split.val_idx), len(self.split.test_idx)) == 0:
            raise ValueError("普通训练要求 train/val/test 均非空")
        self.logger.info(
            f"固定划分: {split_path}, seed={self.split.metadata['seed']}, "
            f"比例={self.split.metadata['ratios']}, sample_id_source=row_index"
        )
        # 只提取训练行；不对完整 phenotype 做统计或建立标签相关指纹。
        train_codes = self.dataset.snp[train_idx]
        train_phenotype = self.dataset.phenotype[train_idx]
        train_dosage = model_codes_to_dosage(train_codes)
        self.train_indices_sha256 = training_indices_hash(train_idx)
        self.screening_cache_key = compute_cache_key(
            self.snp_file,
            self.pheno_dir,
            self.phenotype_names,
            self.screen_k,
            self.gwas_k,
            self.seed,
            split_info=("regular" if self.config.fold_index is None
                        else f"kfold:fold_{self.config.fold_index:02d}")
            + ":" + self.split.metadata["sample_ids_sha256"],
            multivariate=self.screen_multivariate,
            missing_strategy=self.screen_missing_strategy,
            train_indices=train_idx,
            train_genotype=train_codes,
            train_phenotype=train_phenotype,
        )
        screening_imputer = None
        selected = None
        if self.config.preselected_snp_file is not None:
            if self.screen_k or self.gwas_k:
                raise ValueError("preselected_snp_file 不能与内部监督筛选同时启用")
            selected = np.load(self.config.preselected_snp_file, allow_pickle=False)
        elif self.screen_k > 0 or self.gwas_k > 0:
            if self.screen_missing_strategy == "mode":
                screening_imputer = TrainingDosageImputer("mode").fit(
                    train_dosage, train_idx=np.arange(len(train_idx), dtype=np.int64)
                )
                # fit 的输入已是训练子矩阵，同时记录原始行号用于追溯。
                screening_imputer.metadata_["original_train_idx"] = train_idx.tolist()
            selected = screen_snps(
                prepare_screening_dosage(
                    train_dosage,
                    missing_strategy=self.screen_missing_strategy,
                    imputer=screening_imputer,
                ),
                train_phenotype,
                screen_k=self.screen_k,
                gwas_k=self.gwas_k,
                multivariate=self.screen_multivariate,
                cache_key=self.screening_cache_key,
                cache_dir=self.config.screen_cache_dir,
                logger=self.logger,
            )
        self.screened_indices = (
            np.arange(self.original_snp_dim)
            if selected is None
            else np.asarray(selected)
        )
        if (
            self.screened_indices.ndim != 1
            or self.screened_indices.dtype.kind not in "iu"
            or not self.screened_indices.size
            or np.unique(self.screened_indices).size != self.screened_indices.size
            or (self.screened_indices < 0).any()
            or (self.screened_indices >= self.original_snp_dim).any()
        ):
            raise ValueError("筛选缓存/结果不是有效的原始 SNP 列索引")
        self.screened_indices = self.screened_indices.astype(np.int64)
        self.dataset.snp = self.dataset.snp[:, self.screened_indices]
        self.snp_dim = self.dataset.snp.shape[1]

        # Wide 拟合独立于标签，且不将筛选的填补结果写回 Deep 类别矩阵。
        wide_metadata = {"strategy": self.config.wide_missing_strategy}
        if self.config.wide_missing_strategy == "mode":
            wide_imputer = TrainingDosageImputer("mode").fit(
                model_codes_to_dosage(self.dataset.snp), train_idx=train_idx
            )
            self.wide_fill_values = wide_imputer.fill_values_.copy()
            wide_metadata = wide_imputer.metadata_
        else:
            prepare_screening_dosage(train_dosage[:, self.screened_indices])

        self.train_loader, self.val_loader, self.test_loader, self.scalers = (
            get_dataloader(
                self.dataset,
                self.batch_size,
                self.norm_method,
                self.scaler_type,
                device=self.device,
                num_workers=self.num_workers,
                indices=self.split,
            )
        )
        save_scalers(self.scalers, self.scaler_file)
        artifacts = Path(self.config.preprocessing_save_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        copied_split = artifacts / "split.npz"
        if Path(split_path).resolve() != copied_split.resolve():
            shutil.copyfile(split_path, copied_split)
        split_sha256 = hashlib.sha256(copied_split.read_bytes()).hexdigest()
        np.save(artifacts / "screened_indices.npy", self.screened_indices)
        self.preprocessing_metadata = {
            "schema_version": 3,
            "training_mode": "regular" if self.config.fold_index is None else "kfold",
            "fold_index": self.config.fold_index,
            "model_initialization_seed": self.seed,
            "split_source": str(Path(split_path).resolve()),
            "split_copy": "split.npz",
            "split_sha256": split_sha256,
            "split_seed": self.split.metadata["seed"],
            "split_ratios": self.split.metadata["ratios"],
            "sample_id_source": self.split.metadata["sample_id_source"],
            "sample_ids_sha256": self.split.metadata["sample_ids_sha256"],
            "train_indices_sha256": self.train_indices_sha256,
            "original_snp_dim": self.original_snp_dim,
            "snp_dim": self.snp_dim,
            "phenotype_names": self.phenotype_names,
            "time_coordinates": self.config.time_coordinates,
            "time_labels": getattr(self.dataset, "time_labels", None),
            "screened_indices_sha256": hashlib.sha256(
                (artifacts / "screened_indices.npy").read_bytes()
            ).hexdigest(),
            "scaler_sha256": hashlib.sha256(Path(self.scaler_file).read_bytes()).hexdigest(),
            "wide_fill_values_sha256": array_fingerprint(np.asarray(
                [] if self.wide_fill_values is None else self.wide_fill_values,
                dtype=np.float32,
            )),
            "screen_k": self.screen_k,
            "gwas_k": self.gwas_k,
            "screen_multivariate": self.screen_multivariate,
            "screen_missing_strategy": self.screen_missing_strategy,
            "screening_cache_key": self.screening_cache_key,
            "preselected_snp_file": self.config.preselected_snp_file,
            "training_genotype_sha256": array_fingerprint(train_codes),
            "training_phenotype_sha256": array_fingerprint(train_phenotype),
            "screening_dosage_encoding_version": GENOTYPE_ENCODING_VERSION,
            "screening_algorithm_version": SCREENING_ALGORITHM_VERSION,
            "screening_imputer": None
            if screening_imputer is None
            else screening_imputer.metadata_,
            "wide_input_mode": "dosage_v1",
            "wide_imputer": wide_metadata,
        }
        (artifacts / "preprocessing.json").write_text(
            json.dumps(self.preprocessing_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_model_config(self.config, self.best_model_save_dir)
        config_path = self.best_model_save_dir / "model_config.json"
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_config["snp_dim"] = self.snp_dim
        config_path.write_text(
            json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.logger.info(
            f"仅训练个体筛选: train_n={len(train_idx)}, 原始维度={self.original_snp_dim}, "
            f"筛选后={self.snp_dim}, train_hash={self.train_indices_sha256}, "
            f"cache_key={self.screening_cache_key}, Wide策略={self.config.wide_missing_strategy}"
        )

    def _create_model(self):
        """获取模型"""
        return TFPP(
            snp_dim=self.snp_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            num_embeddings=self.num_embeddings,
            time_coordinates=self.time_coordinates,
            phenotype_names=self.phenotype_names,
            dropout=self.dropout,
            degree=self.degree,
            ablation_mode=self.ablation_mode,
            wide_input_mode="dosage_v1",
            wide_fill_values=self.wide_fill_values,
        ).to(self.device)

    def _create_dataset(self):
        """
        获取数据集对象
        """
        return TFPPDataset(self.snp_file, self.pheno_dir, self.phenotype_names)

    def _create_loss(self):
        """根据配置创建损失函数。"""
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "mse_pcc":
            return MSEPCCLoss(pcc_weight=self.pcc_weight)
        elif self.loss_type == "mse_offset":
            return MSEOffsetLoss(
                offset_weight=self.offset_weight,
                pcc_weight=self.pcc_weight,
            )
        else:
            raise ValueError(f"未知 loss_type: {self.loss_type}")

    def _create_optimizer_and_scheduler(self):
        """创建优化器与学习率调度器"""
        base_lr = self.lr
        wide_lr_scale = 0.1
        wide_wd_scale = 1.0

        wide_params = []
        deep_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "wide_layer" in name:
                wide_params.append(param)
            else:
                deep_params.append(param)

        optimizer = optim.AdamW(
            [
                {
                    "params": wide_params,
                    "lr": base_lr * wide_lr_scale,
                    "weight_decay": self.weight_decay * wide_wd_scale,
                },
                {
                    "params": deep_params,
                    "lr": base_lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        if self.use_warmup:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.epochs - self.warmup_epochs),
                eta_min=self.min_lr,
            )
            scheduler = WarmupSequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.min_lr
            )

        return optimizer, scheduler

    def _train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> tuple[float, dict[str, float]]:
        """训练一个 epoch 并返回损失与指标。

        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器。
        optimizer : optim.Optimizer
            优化器。
        criterion : nn.Module
            损失函数。

        Returns
        -------
        tuple[float, dict[str, float]]
            平均训练损失和评估指标字典。
        """
        self.model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        all_preds = []
        all_targets = []

        # 梯度累积: 在循环外清空梯度, 只在达到 accumulation_steps 或最后一个 batch 时才 step
        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            phenotype_pred = self.model(X)
            loss = criterion(phenotype_pred, y)

            # L1 稀疏惩罚: 对宽通路参数施加 L1 正则
            if self.l1_wide_weight > 0 and hasattr(self.model, "wide_layer"):
                l1_penalty = 0.0
                wide_layer = self.model.wide_layer
                if hasattr(wide_layer, "snp_importance"):
                    l1_penalty = l1_penalty + wide_layer.snp_importance.abs().sum()
                if hasattr(wide_layer, "base_linear"):
                    l1_penalty = l1_penalty + wide_layer.base_linear.weight.abs().sum()
                if hasattr(wide_layer, "dynamic_linear"):
                    l1_penalty = (
                        l1_penalty + wide_layer.dynamic_linear.weight.abs().sum()
                    )
                if hasattr(wide_layer, "poly_linear"):
                    l1_penalty = l1_penalty + wide_layer.poly_linear.weight.abs().sum()
                loss = loss + self.l1_wide_weight * l1_penalty

            # 损失按累积步数缩放, 反传到累积梯度上 (保证梯度量级与等效大 batch 一致)
            loss_scaled = loss / self.accumulation_steps
            loss_scaled.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (
                batch_idx + 1
            ) == num_batches:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            # 指标统计使用未缩放的原始 loss, 与历史口径保持一致
            train_loss += loss.item()
            all_preds.append(phenotype_pred.detach())
            all_targets.append(y.detach())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        phenotype_metrics = calculate_phenotype_metrics_by_time(
            all_preds, all_targets, self.phenotype_names
        )
        phenotype_metrics_avg = calculate_metrics_by_average(phenotype_metrics)

        return train_loss / num_batches, phenotype_metrics_avg

    def _validate_epoch(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
        """验证一个 epoch 并返回损失与各表型指标。

        Parameters
        ----------
        val_loader : DataLoader
            验证数据加载器。
        criterion : nn.Module
            损失函数。

        Returns
        -------
        tuple[float, dict[str, dict[str, float]], dict[str, float]]
            平均验证损失、各表型指标字典和综合平均指标字典。
        """
        self.model.eval()
        num_batches = len(val_loader)
        val_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                phenotype_pred = self.model(X)

                # 计算 Loss (只用预测值)
                loss = criterion(phenotype_pred, y)
                val_loss += loss.item()

                # 收集预测值和真实值
                all_preds.append(phenotype_pred)
                all_targets.append(y)

        # 计算整体指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算各表型指标
        phenotype_metrics = calculate_phenotype_metrics_by_time(
            all_preds, all_targets, self.phenotype_names
        )
        phenotype_metrics_avg = calculate_metrics_by_average(phenotype_metrics)

        # 返回值增加了 avg_gate
        return val_loss / num_batches, phenotype_metrics, phenotype_metrics_avg

    def _evaluate_test_set(
        self, test_loader: DataLoader, scalers: dict
    ) -> dict[str, Any]:
        """在测试集上进行最终评估（使用反归一化后的数据）。

        Parameters
        ----------
        test_loader : DataLoader
            测试数据加载器。
        scalers : dict
            归一化器字典。

        Returns
        -------
        dict[str, Any]
            包含归一化和反归一化两组评估结果的字典。
        """
        reservation = Path(self.config.output_dir) / "test_evaluation.json"
        with reservation.open("x", encoding="utf-8") as handle:
            json.dump({"test_evaluation_count": 1, "train_indices_sha256": self.train_indices_sha256}, handle)
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                phenotype_pred = self.model(X)

                all_preds.append(phenotype_pred.cpu())
                all_targets.append(y.cpu())

        # 合并所有批次
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 反归一化
        all_preds_denorm = denormalize_phenotype(all_preds, scalers)
        all_targets_denorm = denormalize_phenotype(all_targets, scalers)

        # 计算归一化数据的指标
        norm_pheno_metrics = calculate_phenotype_metrics_by_time(
            all_preds, all_targets, self.phenotype_names
        )
        norm_phenotype_metrics_avg = calculate_metrics_by_average(norm_pheno_metrics)

        # 计算反归一化数据的指标
        denorm_pheno_metrics = calculate_phenotype_metrics_by_time(
            all_preds_denorm, all_targets_denorm, self.phenotype_names
        )
        denorm_phenotype_metrics_avg = calculate_metrics_by_average(
            denorm_pheno_metrics
        )

        self.test_predictions = {
            "predictions_normalized": all_preds.numpy(),
            "targets_normalized": all_targets.numpy(),
            "predictions": all_preds_denorm.numpy(),
            "targets": all_targets_denorm.numpy(),
        }

        return {
            "normalized": {
                "phenotypes": norm_pheno_metrics,
                "overall": norm_phenotype_metrics_avg,
            },
            "denormalized": {
                "phenotypes": denorm_pheno_metrics,
                "overall": denorm_phenotype_metrics_avg,
            },
        }

    def _monitor_gradients(self):
        """监控梯度统计信息"""
        grad_norms = []
        grad_means = []
        grad_stds = []
        layer_names = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                grad_norms.append(grad_norm)
                grad_means.append(grad_mean)
                grad_stds.append(grad_std)
                layer_names.append(name)

                # 记录异常梯度
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"异常梯度: {name} - 包含NaN或Inf")

        if grad_norms:  # 确保有梯度数据
            # 记录整体统计
            self.logger.info(
                f"梯度范数: 平均={np.mean(grad_norms):.6f}, "
                f"最大={np.max(grad_norms):.6f}, "
                f"最小={np.min(grad_norms):.6f}"
            )

            # 记录梯度最大的前3层
            if len(grad_norms) > 3:
                top_indices = np.argsort(grad_norms)[-3:][::-1]
                for idx in top_indices:
                    self.logger.info(
                        f"  梯度最大层: {layer_names[idx]} - 范数={grad_norms[idx]:.6f}"
                    )

    def _train_regular(self) -> dict[str, Any]:
        """常规训练模式（单次训练验证测试划分）。

        Returns
        -------
        dict[str, Any]
            训练统计信息字典。
        """
        self.logger.info("=" * 60)
        self.logger.info("常规训练")
        self.logger.info("=" * 60)

        # 输出配置信息
        self._log_configuration()

        if (Path(self.config.output_dir) / "test_evaluation.json").exists():
            raise ValueError("test 已预约/评估，禁止重新训练后重测")
        torch.manual_seed(self.seed)  # 训练 RNG 不依赖其他实例或外部探查消耗。
        start_time = time.time()

        # 初始化时已固定划分并只用训练个体完成筛选和 scaler 拟合，不能再次划分。
        train_loader, val_loader, test_loader, scalers = (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scalers,
        )
        self.logger.info("数据加载完成")

        # 创建优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler()

        criterion = self._create_loss()

        # 训练状态跟踪
        best_val_loss = float("inf")
        best_val_metrics = {}
        best_val_pheno_metrics = {}  # 记录各表型的最优指标
        best_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        patience_counter = 0
        best_epoch = -1

        # 训练历史记录
        train_loss_history = []
        val_loss_history = []
        train_metrics_history = []  # 每轮训练的综合指标
        val_metrics_history = []  # 每轮验证的综合指标
        val_pheno_metrics_history = []  # 每轮验证各个表型的详细指标

        # 训练循环
        for epoch in range(self.epochs):
            # 训练一个epoch
            avg_train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, criterion
            )

            # 验证一个epoch
            avg_val_loss, val_pheno_metrics, val_metrics = self._validate_epoch(
                val_loader, criterion
            )

            if not np.isfinite(avg_val_loss):
                raise ValueError("验证损失非有限，禁止保存初始/未选择模型或评估 test")

            # 记录损失历史
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            val_pheno_metrics_history.append(val_pheno_metrics)

            # 日志输出
            current_lr = optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Loss[train:{avg_train_loss:.6f} / val:{avg_val_loss:.6f}] | "
                f"PCC[train:{train_metrics['pcc']:.6f} / val:{val_metrics['pcc']:.6f}] | "
                f"R²[train:{train_metrics['r2']:.6f} / val:{val_metrics['r2']:.6f}] | "
                f"lr:{current_lr:.2e}"
                f"{' ★' if avg_val_loss < best_val_loss else ''}"
            )

            # 学习率调度
            scheduler.step()

            # 更新最佳指标
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_pheno_metrics = val_pheno_metrics  # 各个表型的最优指标
                best_val_metrics = val_metrics  # 所有表型的综合最优指标
                best_epoch = epoch + 1
                patience_counter = 0
                # 记录最佳epoch的各表型指标
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.patience:
                self.logger.info(f"连续 {self.patience} 轮无改善，触发早停！")
                break

        training_time = time.time() - start_time
        final_lr = optimizer.param_groups[0]["lr"]

        # 输出训练结果摘要
        print()
        self.logger.info("训练完成摘要:")
        self.logger.info(f"    best_epoch: {best_epoch}/{self.epochs}")
        self.logger.info(f"    best_val_loss: {best_val_loss:.6f}")
        self.logger.info(f"    best_val_mse: {best_val_metrics['mse']:.6f}")
        self.logger.info(f"    best_val_rmse: {best_val_metrics['rmse']:.6f}")
        self.logger.info(f"    best_val_mae: {best_val_metrics['mae']:.6f}")
        self.logger.info(f"    best_val_pcc: {best_val_metrics['pcc']:.6f}")
        self.logger.info(f"    best_val_r²: {best_val_metrics['r2']:.6f}")
        self.logger.info(f"    final_lr: {final_lr:.2e}")
        self.logger.info(f"    training_time: {training_time / 60:.2f} 分钟\n")

        # 输出各表型的最佳指标
        self.logger.info("各表型最佳验证指标:")
        for pheno_name in self.phenotype_names:
            metrics = best_val_pheno_metrics[pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )
        print()

        # 在测试集上评估
        self.logger.info("=" * 60)
        self.logger.info("测试集评估")
        self.logger.info("=" * 60)

        # 加载最佳模型
        self.model.load_state_dict(best_model_state)

        # 加载scalers
        scalers = load_scalers(self.scaler_file)
        self.logger.info(f"Scaler对象已从 {self.scaler_file} 加载")

        # 测试集评估
        test_results = self._evaluate_test_set(test_loader, scalers)

        # 输出测试集结果
        self.logger.info("（归一化）数据的测试集整体指标:")
        for metric, value in test_results["normalized"]["overall"].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("（反归一化）数据的测试集整体指标:")
        for metric, value in test_results["denormalized"]["overall"].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("（归一化）各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results["normalized"]["phenotypes"][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}，"
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )
        print()

        self.logger.info("（反归一化）各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results["denormalized"]["phenotypes"][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}，"
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )
        print()

        # 保存模型和训练信息
        if self.is_save_training_info:
            self._save_model_info(
                best_model_state,
                best_val_loss,
                best_epoch,
                best_val_metrics,
                best_val_pheno_metrics,
                training_time,
                train_loss_history,
                val_loss_history,
                train_metrics_history,
                val_metrics_history,
                val_pheno_metrics_history,
                final_lr,
                test_results,
            )

        return {
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "training_time": training_time,
            "test_results": test_results,
        }

    def _save_model_info(
        self,
        best_model_state: dict[str, Any] | None,
        best_val_loss: float,
        best_epoch: int,
        best_val_metrics: dict,
        best_val_pheno_metrics: dict[str, dict[str, float]] | None,
        training_time: float,
        train_loss_history: list,
        val_loss_history: list,
        train_metrics_history: list,
        val_metrics_history: list,
        val_pheno_metrics_history: list,
        final_lr: float,
        test_results: dict,
    ) -> None:
        """保存模型权重及完整的训练元数据（采用 PT + JSON + CSV 分离存储）。

        Parameters
        ----------
        best_model_state : dict[str, Any] | None
            最优模型状态字典，为 None 时保存空状态。
        best_val_loss : float
            最优验证损失。
        best_epoch : int
            最优 epoch 编号。
        best_val_metrics : dict
            所有表型的综合最优验证指标。
        best_val_pheno_metrics : dict[str, dict[str, float]] | None
            各表型的最优验证指标。
        training_time : float
            训练总耗时（秒）。
        train_loss_history : list
            每轮训练损失列表。
        val_loss_history : list
            每轮验证损失列表。
        train_metrics_history : list
            每轮训练综合指标列表。
        val_metrics_history : list
            每轮验证综合指标列表。
        val_pheno_metrics_history : list
            每轮验证各表型详细指标列表。
        final_lr : float
            训练结束时的最终学习率。
        test_results : dict
            测试集评估结果。
        """
        # ==========================================
        # 1. 保存纯模型权重 (.pt)
        # ==========================================
        model_path = self.best_model_save_dir / "best_model.pt"
        # 推荐将权重 clone 并转移到 CPU 上，避免绑定 GPU 显存，方便后续在任意机器上加载
        if best_model_state is None:
            cpu_model_state = {}
            self.logger.warning("best_model_state 为 None，已保存空模型状态。")
        else:
            cpu_model_state = {k: v.cpu().clone() for k, v in best_model_state.items()}
        torch.save(cpu_model_state, model_path)
        self.logger.info(f"【模型权重】已保存至: {model_path}")

        # 保存模型结构配置到 models/ 目录 (与 .pt 同级, 供 load_tfpp_model 读取)
        save_model_config(self.config, self.best_model_save_dir)
        _cfg_path = self.best_model_save_dir / "model_config.json"
        if _cfg_path.exists():
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _mc = json.load(_f)
            _mc["snp_dim"] = self.snp_dim
            with open(_cfg_path, "w", encoding="utf-8") as _f:
                json.dump(_mc, _f, ensure_ascii=False, indent=2)

        from utils.ArtifactIntegrity import CHECKPOINT_PROTOCOL, seal_checkpoint
        self.preprocessing_metadata["checkpoint_protocol"] = CHECKPOINT_PROTOCOL
        Path(self.config.preprocessing_save_dir, "preprocessing.json").write_text(
            json.dumps(self.preprocessing_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        seal_checkpoint(self.config.output_dir, extra_files=("test_evaluation.json", "preprocessing/screened_indices.npy"))

        # ==========================================
        # 2. 保存训练摘要与配置元数据 (.json)
        # ==========================================
        summary_info = {
            "best_epoch": int(best_epoch),
            "total_epochs": int(self.epochs),
            "best_val_loss": float(best_val_loss),
            "training_time_seconds": float(training_time),
            "final_lr": float(final_lr) if final_lr is not None else None,
            "best_val_metrics": best_val_metrics,
            "best_val_pheno_metrics": best_val_pheno_metrics,
            "test_results": test_results,
        }

        summary_path = self.training_history_save_dir / "training_summary.json"

        # 使用自定义转换处理可能遗留的 NumPy/PyTorch 数据类型，防止 JSON 序列化报错
        def default_dump(obj):
            if hasattr(obj, "item"):  # 处理 tensor 或 numpy 标量
                return obj.item()
            return str(obj)  # 兜底方案

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                summary_info, f, ensure_ascii=False, indent=4, default=default_dump
            )
        self.logger.info(f"【训练摘要】已保存至: {summary_path}")

        # ==========================================
        # 3. 保存训练过程历史曲线 (.csv)
        # ==========================================
        # 保存综合训练历史为 CSV
        history_df = pd.DataFrame(
            {
                "epoch": range(1, len(train_loss_history) + 1),
                "train_loss": [float(x) for x in train_loss_history],
                "val_loss": [float(x) for x in val_loss_history],
                "train_mse": [m["mse"] for m in train_metrics_history],
                "train_rmse": [m["rmse"] for m in train_metrics_history],
                "train_mae": [m["mae"] for m in train_metrics_history],
                "train_pcc": [m["pcc"] for m in train_metrics_history],
                "train_r2": [m["r2"] for m in train_metrics_history],
                "val_mse": [m["mse"] for m in val_metrics_history],
                "val_rmse": [m["rmse"] for m in val_metrics_history],
                "val_mae": [m["mae"] for m in val_metrics_history],
                "val_pcc": [m["pcc"] for m in val_metrics_history],
                "val_r2": [m["r2"] for m in val_metrics_history],
            }
        )
        history_path = self.training_history_save_dir / "training_history_metrics.csv"
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"【综合历史指标】已保存至: {history_path}")

        # 保存所有轮次的各个表型的所有训练指标为 CSV
        metric_names = ["mse", "rmse", "mae", "pcc", "r2"]
        for metric in metric_names:
            data = {"epoch": list(range(1, len(val_pheno_metrics_history) + 1))}

            # 为每个表型添加该指标的数据
            for phenotype in self.phenotype_names:
                phenotype_data = []
                for epoch_metrics in val_pheno_metrics_history:
                    if (
                        phenotype in epoch_metrics
                        and metric in epoch_metrics[phenotype]
                    ):
                        # 确保提取出来的是标量，避免存入带梯度的 Tensor
                        val = epoch_metrics[phenotype][metric]
                        phenotype_data.append(
                            float(val) if hasattr(val, "item") else val
                        )
                    else:
                        phenotype_data.append(None)
                data[phenotype] = phenotype_data

            df_metric = pd.DataFrame(data)
            metric_file = (
                self.training_history_save_dir / f"phenotype_{metric}_history.csv"
            )
            df_metric.to_csv(metric_file, index=False)
            self.logger.info(f"【各表型 {metric.upper()} 历史】已保存至: {metric_file}")

    def _kfold_splits(self) -> list[IndividualSplit]:
        """外层 KFold 的 test 只评分；outer train 内独立划出早停 val。"""
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        splits = []
        for fold, (outer_train, test_idx) in enumerate(kfold.split(np.arange(self.num_sample)), 1):
            try:
                train_idx, val_idx = train_test_split(
                    outer_train, test_size=self.config.kfold_val_ratio,
                    random_state=self.seed + fold - 1,
                )
            except ValueError as error:
                raise ValueError("折内 train/val 无法划分，请增加样本或调整 kfold_val_ratio") from error
            split = split_from_indices(train_idx, val_idx, test_idx, self.num_sample, seed=self.seed)
            if min(len(split.train_idx), len(split.val_idx), len(split.test_idx)) == 0:
                raise ValueError("每折 train/val/test 均须非空")
            split.metadata.update(
                generator="sklearn_kfold_inner_shuffle_split_v1", fold_index=fold,
                n_folds=self.n_folds, kfold_val_ratio=self.config.kfold_val_ratio,
                validation_seed=self.seed + fold - 1,
            )
            splits.append(split)
        tests = np.concatenate([split.test_idx for split in splits])
        if not np.array_equal(np.sort(tests), np.arange(self.num_sample)):
            raise ValueError("外层测试个体必须恰好覆盖每个样本一次")
        return splits

    def _prepare_fold(self, fold_index: int, split: IndividualSplit) -> "TFPPTrainer":
        """实例化独立 TFPPTrainer，复用普通训练的预处理、双视图和模型创建顺序。"""
        if (not 1 <= fold_index <= self.n_folds
                or split.metadata.get("fold_index") != fold_index):
            raise ValueError("fold_index 必须与本折 split 一致且在外层折数范围内")
        options = self.config.model_dump()
        options.update(
            exp_name=f"{self.exp_name}/folds/fold_{fold_index:02d}",
            use_kfold=False, fold_index=fold_index, split_file=None,
            seed=self.seed + fold_index, is_save_training_info=True,
        )
        config = type(self.config).model_validate(options)
        split_path = Path(config.preprocessing_save_dir) / "split.npz"
        if split_path.exists():
            old = load_split(split_path, n_samples=self.num_sample)
            if (any(not np.array_equal(getattr(old, key), getattr(split, key))
                    for key in ("train_idx", "val_idx", "test_idx"))
                    or any(old.metadata.get(key) != split.metadata.get(key) for key in
                           ("fold_index", "n_folds", "kfold_val_ratio", "seed", "validation_seed"))):
                raise ValueError(f"已有折划分与当前配置不一致，请使用新实验目录: {split_path}")
        else:
            save_split(split, split_path)
        config = config.model_copy(update={"split_file": str(split_path)})
        # 每折独立初始化；上一折的训练轮数不会影响下一折的随机初态。
        torch.manual_seed(config.seed)
        return TFPPTrainer(config)

    def _save_fold_predictions(self) -> pd.DataFrame:
        """保存最终测试的同一遍输出，携带原始 sample_id 与折号。"""
        test_idx = self.split.test_idx
        sample_ids = np.asarray(self.split.metadata["sample_ids"])[test_idx]
        fold = self.config.fold_index
        directory = Path(self.config.output_dir)
        np.savez_compressed(
            directory / "test_predictions.npz", sample_idx=test_idx, sample_id=sample_ids,
            fold=np.full(len(test_idx), fold, dtype=np.int64),
            phenotype_names=np.asarray(self.phenotype_names),
            time_coordinates=np.asarray(self.time_coordinates), **self.test_predictions,
        )
        rows = []
        for i, idx in enumerate(test_idx):
            for p, name in enumerate(self.phenotype_names):
                row = {"sample_idx": int(idx), "sample_id": sample_ids[i],
                       "fold": fold, "phenotype": name}
                for t in range(len(self.time_coordinates)):
                    row[f"T{t + 1}_Pred"] = self.test_predictions["predictions"][i, p, t]
                    row[f"T{t + 1}_True"] = self.test_predictions["targets"][i, p, t]
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(directory / "test_predictions.csv", index=False)
        return frame

    def _train_with_kfold(self) -> dict[str, Any]:
        """逐折运行完整独立训练；汇总折指标分布，不用混合 PCC 替代各折表现。"""
        self._log_configuration()
        if not self.is_save_training_info:
            self.logger.info("K折要求可复现产物，每折仍保存 checkpoint、预处理和预测。")
        splits = self._kfold_splits()
        fold_results, prediction_frames = [], []
        all_metrics = {key: [] for key in ("mse", "rmse", "mae", "pcc", "r2")}
        pheno_metrics = {name: {key: [] for key in all_metrics} for name in self.phenotype_names}
        start = time.time()
        for fold, split in enumerate(splits, 1):
            self.logger.info(f"Fold {fold}/{self.n_folds}: Train={len(split.train_idx)}, "
                             f"Val={len(split.val_idx)}, Test={len(split.test_idx)}")
            trainer = self._prepare_fold(fold, split)
            try:
                result = trainer.train()  # 只用 val 选模型，恢复最佳权重后最终评估 test 一次。
                frame = trainer._save_fold_predictions()
                prediction_frames.append(frame)
                result.update(fold=fold, fold_dir=trainer.config.output_dir,
                              train_indices_sha256=trainer.train_indices_sha256,
                              screening_cache_key=trainer.screening_cache_key,
                              snp_dim=trainer.snp_dim)
                (Path(trainer.config.output_dir) / "metrics.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                (Path(trainer.config.output_dir) / "config.json").write_text(
                    trainer.config.model_dump_json(indent=2), encoding="utf-8",
                )
                fold_results.append(result)
                for key in all_metrics:
                    all_metrics[key].append(result["test_results"]["denormalized"]["overall"][key])
                    for name in pheno_metrics:
                        pheno_metrics[name][key].append(
                            result["test_results"]["denormalized"]["phenotypes"][name][key])
            finally:
                for handler in list(trainer.logger.logger.handlers):
                    handler.close()
                    trainer.logger.logger.removeHandler(handler)
                del trainer
        def stats(metrics):
            output = {}
            for key, values in metrics.items():
                valid = np.asarray(values)[np.isfinite(values)]
                output[key] = {"mean": float(valid.mean()) if len(valid) else float("nan"),
                               "std": float(valid.std(ddof=0)) if len(valid) else float("nan"),
                               "valid_count": int(len(valid)), "values": values}
            return output
        summary = {
            "metrics_protocol": "time_adjusted_r2_nan_pcc_v2",
            "n_folds": self.n_folds, "outer_seed": self.seed,
            "kfold_val_ratio": self.config.kfold_val_ratio,
            "fold_results": fold_results, "overall_denorm_metrics": all_metrics,
            "phenotype_denorm_metrics": pheno_metrics,
            "overall_statistics": stats(all_metrics),
            "phenotype_statistics": {name: stats(values) for name, values in pheno_metrics.items()},
            "std_ddof": 0, "total_time": time.time() - start,
        }
        directory = self.training_history_save_dir
        (directory / "kfold_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        pd.DataFrame([
            {"fold": result["fold"], "best_epoch": result["best_epoch"],
             "best_val_loss": result["best_val_loss"],
             **result["test_results"]["denormalized"]["overall"]}
            for result in fold_results
        ]).to_csv(directory / "kfold_fold_details.csv", index=False)
        pd.DataFrame([
            {"Metric": key, "Mean": values["mean"], "Std": values["std"]}
            for key, values in summary["overall_statistics"].items()
        ]).to_csv(directory / "kfold_overall_stats.csv", index=False)
        pd.DataFrame([
            {"Phenotype": name, "Metric": key, "Mean": values["mean"], "Std": values["std"]}
            for name, metrics in summary["phenotype_statistics"].items() for key, values in metrics.items()
        ]).to_csv(directory / "kfold_phenotype_stats.csv", index=False)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.sort_values(["sample_idx", "phenotype"], kind="stable")
        predictions.to_csv(directory / "kfold_predictions.csv", index=False)
        for key, values in summary["overall_statistics"].items():
            self.logger.info(f"K折 {key.upper()}: {values['mean']:.6f} ± {values['std']:.6f}")
        return summary

    def train(self) -> dict[str, Any]:
        """根据配置选择训练模式（K折交叉验证或常规训练）。

        Returns
        -------
        dict[str, Any]
            训练统计信息字典。
        """
        if self.use_kfold:
            # 使用K折交叉验证
            return self._train_with_kfold()
        else:
            # 使用常规训练
            return self._train_regular()

    def _log_configuration(self) -> None:
        """记录配置信息"""
        self.logger.info("训练配置:")
        self.logger.info(f"    实验名称: {self.exp_name}")
        self.logger.info(f"    样本数量: {self.num_sample}")
        self.logger.info(f"    特征数量 (snp_dim): {self.snp_dim}")
        self.logger.info(
            f"    表型数量: {len(self.phenotype_names)} {self.phenotype_names}"
        )
        self.logger.info(f"    时间点数: {self.time_coordinates}")
        self.logger.info(
            f"    训练模式: {'K折交叉验证' if self.use_kfold else '常规训练'}"
        )
        if self.use_kfold:
            self.logger.info(f"    折数: {self.n_folds}")
            self.logger.info(f"    外层 train 内验证比例: {self.config.kfold_val_ratio}")
        else:
            self.logger.info(f"    train_ratio: {self.train_ratio}")
            self.logger.info(f"    val_ratio: {self.val_ratio}")
            self.logger.info(f"    test_ratio: {self.test_ratio}")

        self.logger.info("-" * 60)
        self.logger.info("训练超参数:")
        self.logger.info(f"    epochs: {self.epochs}")
        self.logger.info(f"    batch_size: {self.batch_size}")
        self.logger.info(f"    accumulation_steps: {self.accumulation_steps}")
        self.logger.info(f"    等效批量: {self.batch_size * self.accumulation_steps}")
        self.logger.info(f"    lr: {self.lr}")
        self.logger.info(f"    min_lr: {self.min_lr}")
        self.logger.info(f"    weight_decay: {self.weight_decay}")
        self.logger.info(f"    dropout: {self.dropout}")
        self.logger.info(f"    grad_clip_norm: {self.grad_clip_norm}")
        self.logger.info(f"    patience: {self.patience}")
        self.logger.info(f"    seed: {self.seed}")
        self.logger.info(f"    device: {self.device}")
        self.logger.info(f"    num_workers: {self.num_workers}")

        self.logger.info("-" * 60)
        self.logger.info("学习率调度器:")
        self.logger.info(f"    use_warmup: {self.use_warmup}")
        if self.use_warmup:
            self.logger.info(f"    warmup_epochs: {self.warmup_epochs}")
            self.logger.info(f"    warmup_start_factor: {self.warmup_start_factor}")
            self.logger.info(f"    warmup_end_factor: {self.warmup_end_factor}")

        self.logger.info("-" * 60)
        self.logger.info("损失函数与正则化:")
        self.logger.info(f"    loss_type: {self.loss_type}")
        if self.loss_type in ("mse_pcc", "mse_offset"):
            self.logger.info(f"    pcc_weight: {self.pcc_weight}")
        if self.loss_type == "mse_offset":
            self.logger.info(f"    offset_weight: {self.offset_weight}")
        self.logger.info(f"    l1_wide_weight: {self.l1_wide_weight}")

        self.logger.info("-" * 60)
        self.logger.info("数据预处理:")
        self.logger.info(f"    norm_method: {self.norm_method}")
        self.logger.info(f"    scaler_type: {self.scaler_type}")
        self.logger.info(f"    screen_k: {self.screen_k}")
        self.logger.info(f"    gwas_k: {self.gwas_k}")
        self.logger.info(f"    screen_multivariate: {self.screen_multivariate}")
        self.logger.info(f"    screen_missing_strategy: {self.screen_missing_strategy}")

        self.logger.info("-" * 60)
        self.logger.info("模型配置:")
        self.logger.info(f"    snp_dim: {self.snp_dim}")
        self.logger.info(f"    d_model: {self.d_model}")
        self.logger.info(f"    num_heads: {self.num_heads}")
        self.logger.info(f"    num_layers: {self.num_layers}")
        self.logger.info(f"    d_ff: {self.d_ff}")
        self.logger.info(f"    num_embeddings: {self.num_embeddings}")
        self.logger.info(f"    degree: {self.degree}")
        self.logger.info(f"    ablation_mode: {self.ablation_mode}")
        self.logger.info(f"    dropout: {self.dropout}")

        self.logger.info("-" * 60)
        self.logger.info("路径配置:")
        self.logger.info(f"    snp_file: {self.snp_file}")
        self.logger.info(f"    pheno_dir: {self.pheno_dir}")
        self.logger.info(f"    model_save_dir: {self.best_model_save_dir}")
        self.logger.info(
            f"    training_history_save_dir: {self.training_history_save_dir}"
        )
        self.logger.info(f"    scaler_save_dir: {self.scaler_save_dir}")
        self.logger.info(f"    log_file: {self.log_file}")
        self.logger.info("=" * 60)
