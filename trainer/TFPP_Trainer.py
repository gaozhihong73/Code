import json
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
from dataloader import TFPPDataset, get_dataloader, get_dataloader_with_kfold
from model import TFPP
from trainer._scheduler import WarmupSequentialLR
from utils import (
    Logger,
    MSEPCCLoss,
    calculate_metrics_by_average,
    calculate_phenotype_metrics_by_time,
    denormalize_phenotype,
    load_scalers,
    save_scalers,
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
        self.snp_dim: int = self.dataset.snp.shape[1]
        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio
        self.norm_method = config.norm_method
        self.scaler_type = config.scaler_type

        # 日志记录器
        self.log_file = self.config.log_file
        self.logger = Logger(self.log_file, unique=False)

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
        self.seed = config.seed
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

        # 模型配置
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.num_embeddings = config.num_embeddings
        self.d_ff = config.d_ff
        self.time_coordinates = config.time_coordinates

        # 模型初始化
        self.model = self._create_model()

        # K折交叉验证参数
        self.use_kfold = config.use_kfold
        self.n_folds = config.n_folds

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
        ).to(self.device)

    def _create_dataset(self):
        """
        获取数据集对象
        """
        return TFPPDataset(self.snp_file, self.pheno_dir, self.phenotype_names)

    def _create_optimizer_and_scheduler(self):
        """
        创建优化器：实现分层学习率 + 分层权重衰减
        """
        base_lr = self.lr

        # 建议：如果 Config 支持，最好将这两个比例也放到 config 中
        wide_lr_scale = 0.1
        wide_wd_scale = 100.0

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

        # 安全设定：多组不同 LR 时，安全起见 eta_min 设为 0，防止小 LR 组反向放大
        safe_eta_min = 0.0
        if self.use_warmup:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=self.epochs - self.warmup_epochs, eta_min=safe_eta_min
            )
            # 使用 WarmupSequentialLR 将 warmup 和 main_scheduler 无缝拼接
            # (本仓库自带的轻量 wrapper, 与 SequentialLR 行为等价, 但避免上游 EPOCH_DEPRECATION_WARNING)
            scheduler = WarmupSequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=safe_eta_min
            )

        # 改进：无论是否有 warmup，都只返回 optimizer 和一个统一的 scheduler
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
        self.logger.info("=" * 80)
        self.logger.info("常规训练")
        self.logger.info("=" * 80)

        # 输出配置信息
        self._log_configuration()

        start_time = time.time()

        # 创建数据加载器
        train_loader, val_loader, test_loader, scalers = get_dataloader(
            dataset=self.dataset,
            norm_method=self.norm_method,
            scaler_type=self.scaler_type,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
            device=self.device,
        )

        # 保存表型的归一化器，用于在预测的时候进行反归一化
        save_scalers(scalers, self.scaler_file)
        self.logger.info(f"归一化器已保存至: {self.scaler_file}")
        self.logger.info("数据加载完成")

        # 创建优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler()

        criterion = MSEPCCLoss()

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
        self.logger.info("=" * 80)
        self.logger.info("测试集评估")
        self.logger.info("=" * 80)

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

    def _train_single_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        scalers: dict,
    ) -> dict[str, Any]:
        """训练单个折。

        Parameters
        ----------
        fold_idx : int
            当前折的索引。
        train_loader : DataLoader
            训练数据加载器。
        val_loader : DataLoader
            验证数据加载器。
        test_loader : DataLoader
            测试数据加载器。
        scalers : dict
            归一化器字典。

        Returns
        -------
        dict[str, Any]
            该折的训练结果字典。
        """
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"开始训练第 {fold_idx + 1}/{self.n_folds} 折")
        self.logger.info(f"{'=' * 80}")

        # 重新初始化模型（每折使用新模型）
        self.model = self._create_model()

        # 创建优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler()
        criterion = MSEPCCLoss()

        # 训练状态跟踪
        best_val_loss = float("inf")
        best_val_metrics = {}
        best_pheno_metrics = {}
        best_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        patience_counter = 0
        best_epoch = -1

        # 训练历史记录
        train_loss_history = []
        val_loss_history = []
        train_metrics_history = []
        val_metrics_history = []

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

            # 记录损失历史
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)

            # 日志输出
            current_lr = optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Loss[train:{avg_train_loss:.6f} / val:{avg_val_loss:.6f}] | "
                f"PCC[train:{train_metrics['pcc']:.6f} / val:{val_metrics['pcc']:.6f}] | "
                f"R²[train:{train_metrics['r2']:.6f} / val:{val_metrics['r2']:.6f}] | "
                f"lr: {current_lr:.2e}"
                f"{' ★' if avg_val_loss < best_val_loss else ''}"
            )

            # 学习率调度
            scheduler.step()

            # 更新最佳指标
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                patience_counter = 0
                best_pheno_metrics = val_pheno_metrics
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.patience:
                self.logger.info(f"连续 {self.patience} 轮无改善，触发早停！")
                break

        # 输出该折训练结果摘要
        self.logger.info(f"第 {fold_idx + 1} 折训练完成摘要:")
        self.logger.info(f"    best_epoch: {best_epoch}/{self.epochs}")
        self.logger.info(f"    best_val_loss: {best_val_loss:.6f}")
        self.logger.info(f"    best_val_mse: {best_val_metrics['mse']:.6f}")
        self.logger.info(f"    best_val_rmse: {best_val_metrics['rmse']:.6f}")
        self.logger.info(f"    best_val_pcc: {best_val_metrics['pcc']:.6f}")
        self.logger.info(f"    best_val_r²: {best_val_metrics['r2']:.6f}\n")

        self.logger.info(
            f"{'=' * 14} 开始测试第 {fold_idx + 1}/{self.n_folds} 折 {'=' * 14}"
        )

        # 重要：这里的 val_loader 实际上是该折的测试集
        # 使用反归一化后的数据进行最终评估
        test_results = self._evaluate_test_set(test_loader, scalers)

        # 输出测试集结果
        self.logger.info("归一化数据的测试集整体指标:")
        for metric, value in test_results["normalized"]["overall"].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("反归一化数据的测试集整体指标:")
        for metric, value in test_results["denormalized"]["overall"].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("归一化各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results["normalized"]["phenotypes"][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )
        print()

        self.logger.info("反归一化各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results["denormalized"]["phenotypes"][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )
        print()

        return {
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "best_model_state": best_model_state,
            "best_pheno_metrics": best_pheno_metrics,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
            "test_results": test_results,  # 包含反归一化后的评估结果
        }

    def _train_with_kfold(self) -> dict[str, Any]:
        """
        使用K折交叉验证评估模型性能

        流程说明：
        1. 将所有数据分成K折
        2. 每折训练一个独立模型
        3. 每折的验证集在训练时用于早停，训练后作为测试集评估
        4. 汇总K折结果，报告反归一化指标的 Mean ± Std

        注意：
        - 这里的"验证集"既用于训练中的模型选择，也用于最终评估
        - 由于每折训练独立模型，不存在信息泄露问题
        - 最终指标全部基于反归一化后的数据

        Returns
        -------
        dict[str, Any]
            K折交叉验证的统计结果字典。
        """
        self._log_configuration()

        # 获取所有样本索引
        indices = np.arange(self.num_sample)

        # 创建K折分割器
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # 存储每折的结果
        fold_results = []

        # 存储所有折的反归一化指标
        all_fold_denorm_metrics = {
            "mse": [],
            "rmse": [],
            "mae": [],
            "pcc": [],
            "r2": [],
        }
        # 存储每个表型在所有折上的反归一化指标
        phenotype_fold_metrics = {
            pheno: {"mse": [], "rmse": [], "mae": [], "pcc": [], "r2": []}
            for pheno in self.phenotype_names
        }
        total_start_time = time.time()

        # K折交叉验证循环
        # 注意：sklearn 的 split 返回的是 (train_index, test_index)
        # 我们将第二个返回值重命名为 test_indices，作为最终评估用的【测试集】
        for fold_idx, (train_val_indices, test_indices) in enumerate(
            kfold.split(indices)
        ):
            # ======================================================
            # 核心修改：二次划分 (从 K-1 折数据中分出验证集)
            # ======================================================
            # 从 train_total_indices 中划分出 10% 作为验证集 (用于 Early Stopping)
            val_relative_ratio = 1.0 / (self.n_folds - 1)
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=val_relative_ratio, random_state=self.seed
            )
            self.logger.info(
                f"Fold {fold_idx + 1}/{self.n_folds}: "
                f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
            )
            # ======================================================
            # 获取 DataLoader (需要修改该函数以接收三个索引列表)
            # ======================================================
            train_loader, val_loader, test_loader, scalers = get_dataloader_with_kfold(
                train_indices=list(train_indices),
                val_indices=list(val_indices),
                test_indices=list(test_indices),
                norm_method=self.norm_method,
                scaler_type=self.scaler_type,
                dataset=self.dataset,
                batch_size=self.batch_size,
                device=self.device,
            )
            # 保存表型的归一化器，用于在预测的时候进行反归一化
            phenotype_scaler_save_path = (
                self.scaler_save_dir / f"scaler_fold_{fold_idx + 1}.pkl"
            )
            save_scalers(scalers, phenotype_scaler_save_path)
            self.logger.info(f"表型归一化器已保存至: {phenotype_scaler_save_path}")

            # 训练当前折
            fold_result = self._train_single_fold(
                fold_idx, train_loader, val_loader, test_loader, scalers
            )
            fold_results.append(fold_result)

            # 收集反归一化后的指标
            denorm_overall = fold_result["test_results"]["denormalized"]["overall"]
            all_fold_denorm_metrics["mse"].append(denorm_overall["mse"])
            all_fold_denorm_metrics["rmse"].append(denorm_overall["rmse"])
            all_fold_denorm_metrics["mae"].append(denorm_overall["mae"])
            all_fold_denorm_metrics["pcc"].append(denorm_overall["pcc"])
            all_fold_denorm_metrics["r2"].append(denorm_overall["r2"])

            # 收集每个表型的反归一化指标
            denorm_phenotypes = fold_result["test_results"]["denormalized"][
                "phenotypes"
            ]
            for pheno_name in self.phenotype_names:
                phenotype_fold_metrics[pheno_name]["mse"].append(
                    denorm_phenotypes[pheno_name]["mse"]
                )
                phenotype_fold_metrics[pheno_name]["rmse"].append(
                    denorm_phenotypes[pheno_name]["rmse"]
                )
                phenotype_fold_metrics[pheno_name]["mae"].append(
                    denorm_phenotypes[pheno_name]["mae"]
                )
                phenotype_fold_metrics[pheno_name]["pcc"].append(
                    denorm_phenotypes[pheno_name]["pcc"]
                )
                phenotype_fold_metrics[pheno_name]["r2"].append(
                    denorm_phenotypes[pheno_name]["r2"]
                )

        total_time = time.time() - total_start_time

        # ========== 输出K折交叉验证总结 ==========
        self.logger.info("=" * 80)
        self.logger.info(f"{self.n_folds}折交叉验证完成!")
        self.logger.info("=" * 80)
        self.logger.info(f"总耗时: {total_time / 60:.2f} 分钟")
        self.logger.info(f"平均每折耗时: {total_time / self.n_folds / 60:.2f} 分钟")
        print()

        # 计算并输出整体反归一化指标的统计
        self.logger.info("整体反归一化指标统计 (Mean ± Std):")
        for metric_name, values in all_fold_denorm_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            self.logger.info(
                f"    {metric_name.upper():>5}: {mean_val:.6f} ± {std_val:.6f}"
            )
        print()

        # 输出每个表型的反归一化指标统计
        self.logger.info("各表型反归一化指标统计 (Mean ± Std):")
        for pheno_name in self.phenotype_names:
            self.logger.info(f"  {pheno_name:>5}:")
            for metric_name, values in phenotype_fold_metrics[pheno_name].items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                self.logger.info(
                    f"      {metric_name.upper():>5}: {mean_val:.6f} ± {std_val:.6f}"
                )
        print()

        # 保存K折统计结果
        if self.is_save_training_info:
            self._save_kfold_results(
                fold_results,
                all_fold_denorm_metrics,
                phenotype_fold_metrics,
                total_time,
            )

        return {
            "fold_results": fold_results,
            "overall_denorm_metrics": all_fold_denorm_metrics,
            "phenotype_denorm_metrics": phenotype_fold_metrics,
            "total_time": total_time,
        }

    def _save_kfold_results(
        self,
        fold_results: list,
        overall_metrics: dict,
        phenotype_metrics: dict,
        total_time: float,
    ):
        """保存 K 折交叉验证结果。

        Parameters
        ----------
        fold_results : list
            每折的详细结果列表。
        overall_metrics : dict
            整体反归一化指标。
        phenotype_metrics : dict
            各表型反归一化指标。
        total_time : float
            总训练时间（秒）。
        """
        # 保存整体指标统计
        overall_stats = pd.DataFrame(
            {
                "Metric": list(overall_metrics.keys()),
                "Mean": [np.mean(v) for v in overall_metrics.values()],
                "Std": [np.std(v) for v in overall_metrics.values()],
                "Min": [np.min(v) for v in overall_metrics.values()],
                "Max": [np.max(v) for v in overall_metrics.values()],
            }
        )
        overall_stats_path = self.training_history_save_dir / "kfold_overall_stats.csv"
        overall_stats.to_csv(overall_stats_path, index=False)
        self.logger.info(f"整体K折统计已保存: {overall_stats_path}\n")

        # 保存各表型指标统计
        phenotype_stats_list = []
        for pheno_name in self.phenotype_names:
            for metric_name, values in phenotype_metrics[pheno_name].items():
                phenotype_stats_list.append(
                    {
                        "Phenotype": pheno_name,
                        "Metric": metric_name.upper(),
                        "Mean": np.mean(values),
                        "Std": np.std(values),
                        "Min": np.min(values),
                        "Max": np.max(values),
                    }
                )
        phenotype_stats = pd.DataFrame(phenotype_stats_list)
        phenotype_stats_path = (
            self.training_history_save_dir / "kfold_phenotype_stats.csv"
        )
        phenotype_stats.to_csv(phenotype_stats_path, index=False)
        self.logger.info(f"各表型K折统计已保存: {phenotype_stats_path}\n")

        # 保存每折的详细结果和模型
        fold_details = []
        for i, result in enumerate(fold_results):
            denorm_overall = result["test_results"]["denormalized"]["overall"]
            fold_details.append(
                {
                    "Fold": i + 1,
                    "Best_Epoch": result["best_epoch"],
                    "Val_Loss": result["best_val_loss"],
                    "Val_MSE": denorm_overall["mse"],
                    "Val_RMSE": denorm_overall["rmse"],
                    "Val_MAE": denorm_overall["mae"],
                    "Val_PCC": denorm_overall["pcc"],
                    "Val_R2": denorm_overall["r2"],
                }
            )
            # 获取最优模型
            best_model_state = result["best_model_state"]
            # 最优模型保存路径
            model_path = self.best_model_save_dir / f"tfpp_best_model_kfold_{i + 1}.pt"
            torch.save(best_model_state, model_path)
        self.logger.info(f"每折最优模型结果已保存:{self.best_model_save_dir}\n")
        fold_details_df = pd.DataFrame(fold_details)
        fold_details_path = self.training_history_save_dir / "kfold_fold_details.csv"
        fold_details_df.to_csv(fold_details_path, index=False)
        self.logger.info(f"每折详细结果已保存: {fold_details_path}\n")

        # 保存K折摘要信息
        summary_path = self.training_history_save_dir / "kfold_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.n_folds}折交叉验证摘要\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"总训练时间: {total_time / 60:.2f} 分钟\n")
            f.write(f"平均每折时间: {total_time / self.n_folds / 60:.2f} 分钟\n\n")

            f.write("整体反归一化指标 (Mean ± Std):\n")
            for metric_name, values in overall_metrics.items():
                f.write(
                    f"  {metric_name.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f}\n"
                )

            f.write("\n各表型反归一化指标 (Mean ± Std):\n")
            for pheno_name in self.phenotype_names:
                f.write(f"  {pheno_name:>5}:\n")
                for metric_name, values in phenotype_metrics[pheno_name].items():
                    f.write(
                        f"    {metric_name.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f}\n"
                    )

        self.logger.info(f"K折摘要已保存: {summary_path}\n")

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
        self.logger.info(f"    特征数量: {self.snp_dim}")
        self.logger.info(
            f"    表型数量: {len(self.phenotype_names)} {self.phenotype_names}"
        )
        self.logger.info(f"    时间点数: {self.time_coordinates}")
        self.logger.info(
            f"    训练模式: {'K折交叉验证' if self.use_kfold else '常规训练'}"
        )
        if self.use_kfold:
            self.logger.info(f"    折数: {self.n_folds}")
        else:
            self.logger.info(f"    test_ratio: {self.test_ratio}")
            self.logger.info(f"    val_ratio: {self.val_ratio}")
        self.logger.info(f"    epochs: {self.epochs}")
        self.logger.info(f"    lr: {self.lr}")
        self.logger.info(f"    weight_decay: {self.weight_decay}")
        self.logger.info(f"    batch_size: {self.batch_size}")
        self.logger.info(f"    patience: {self.patience}")
        self.logger.info(f"    grad_clip_norm: {self.grad_clip_norm}")
        self.logger.info(f"    seed: {self.seed}")
        self.logger.info(f"    device: {self.device}")
        self.logger.info(f"    use_warmup: {self.use_warmup}")
        self.logger.info(f"    norm_method: {self.norm_method}")
        self.logger.info(f"    scaler_type: {self.scaler_type}")
        if self.use_warmup:
            self.logger.info(f"    warmup_epochs: {self.warmup_epochs}")
        self.logger.info("模型配置:")
        self.logger.info(f"    snp_dim: {self.snp_dim}")
        self.logger.info(f"    d_model: {self.d_model}")
        self.logger.info(f"    num_heads: {self.num_heads}")
        self.logger.info(f"    num_layers: {self.num_layers}")
        self.logger.info(f"    num_embeddings: {self.num_embeddings}")
        self.logger.info(f"    d_ff: {self.d_ff}")
        self.logger.info(f"    dropout: {self.dropout}")
        self.logger.info("路径配置:")
        self.logger.info(f"    snp_file: {self.snp_file}")
        self.logger.info(f"    pheno_dir: {self.pheno_dir}")
        self.logger.info(f"    model_save_dir: {self.best_model_save_dir}")
        self.logger.info(
            f"    training_history_save_dir: {self.training_history_save_dir}"
        )
        self.logger.info(f"    phenotype_scaler_dir: {self.scaler_save_dir}")
