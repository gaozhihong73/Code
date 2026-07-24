"""模拟数据 TFPP 训练器

与 TFPPTrainer 的关键差异:
- 使用 SimTFPPDataset (预划分 csv) 代替 TFPPDataset (单 npy + 按 ratio 划分)
- 使用 get_sim_dataloader (不再 random_split / 不再 KFold)
- train() 仅走常规训练分支 _train_regular
- 删除 k-fold 全套方法 (_train_with_kfold / _train_single_fold / _save_kfold_results)
- _log_configuration 路径展示改为 6 个 csv 路径

源文件 TFPP_Trainer.py 保持不变。
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from config import SimTFPPConfig
from dataloader import SimTFPPDataset, get_sim_dataloader
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
    save_scalers,
)


class SimTFPPTrainer:
    """模拟数据 TFPP 训练器 (常规训练, 无 KFold)"""

    def __init__(self, config: SimTFPPConfig):
        # 保存配置
        self.config = config

        # 实验标识
        self.exp_name = config.exp_name
        self.exp_group = config.exp_group
        self.rep_name = config.rep_name

        # 输出目录
        self.best_model_save_dir = Path(config.best_model_save_dir)
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
        self.training_history_save_dir = Path(config.training_history_save_dir)
        self.training_history_save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_save_dir = Path(config.scaler_save_dir)
        self.scaler_save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_file = str(self.scaler_save_dir / "scaler.pkl")

        # 数据集初始化 (模拟数据已预划分, 不需要 train/val/test_ratio 与 snp_file)
        self.phenotype_names = config.phenotype_names
        self.dataset = self._create_dataset()
        self.num_sample = len(self.dataset)
        self.snp_dim: int = self.dataset.snp_train.shape[1]

        # 日志记录器
        self.log_file = config.log_file
        self.logger = Logger(self.log_file, unique=False)

        # 训练超参数
        self.epochs = config.epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        # 梯度累积步数: 固定等效批量为 32, 即 accumulation_steps = 32 / batch_size
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

        # 归一化
        self.norm_method = config.norm_method
        self.scaler_type = config.scaler_type

        # 损失函数
        self.loss_type = config.loss_type
        self.offset_weight = config.offset_weight
        self.pcc_weight = config.pcc_weight
        self.l1_wide_weight = getattr(config, "l1_wide_weight", 1e-5)

        # 模型配置
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.num_embeddings = config.num_embeddings
        self.d_ff = config.d_ff
        self.degree = config.degree
        self.time_coordinates = config.time_coordinates

        # 模型初始化
        self.model = self._create_model()

    def _create_model(self):
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
            ablation_mode=self.config.ablation_mode,
        ).to(self.device)

    def _create_dataset(self):
        return SimTFPPDataset(
            train_snp_csv=self.config.train_snp_csv,
            train_pheno_csv=self.config.train_pheno_csv,
            val_snp_csv=self.config.val_snp_csv,
            val_pheno_csv=self.config.val_pheno_csv,
            test_snp_csv=self.config.test_snp_csv,
            test_pheno_csv=self.config.test_pheno_csv,
            phenotype_names=self.phenotype_names,
            screen_k=self.config.screen_k,
            gwas_k=self.config.gwas_k,
        )

    def _create_optimizer_and_scheduler(self):
        base_lr = self.lr
        wide_lr_scale = 1.0
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

    def _create_loss(self):
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

    def _train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        all_preds = []
        all_targets = []

        # 梯度累积: 在循环外清空梯度
        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            phenotype_pred = self.model(X)
            loss = criterion(phenotype_pred, y)

            # ==========================================================
            # [更新] L1 稀疏惩罚: 针对 EnhancedLegendreWideLayer 进行适配
            # ==========================================================
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

                # 兼容旧版
                if hasattr(wide_layer, "poly_linear"):
                    l1_penalty = l1_penalty + wide_layer.poly_linear.weight.abs().sum()

                # 将 L1 惩罚加入总 Loss
                loss = loss + self.l1_wide_weight * l1_penalty
            # ==========================================================

            # 损失按累积步数缩放
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

    def _validate_epoch(self, val_loader, criterion):
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
                loss = criterion(phenotype_pred, y)
                val_loss += loss.item()
                all_preds.append(phenotype_pred)
                all_targets.append(y)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        phenotype_metrics = calculate_phenotype_metrics_by_time(
            all_preds, all_targets, self.phenotype_names
        )
        phenotype_metrics_avg = calculate_metrics_by_average(phenotype_metrics)

        return val_loss / num_batches, phenotype_metrics, phenotype_metrics_avg

    def _evaluate_test_set(self, test_loader, scalers):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                phenotype_pred = self.model(X)
                all_preds.append(phenotype_pred.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        all_preds_denorm = denormalize_phenotype(all_preds, scalers)
        all_targets_denorm = denormalize_phenotype(all_targets, scalers)

        norm_pheno_metrics = calculate_phenotype_metrics_by_time(
            all_preds, all_targets, self.phenotype_names
        )
        norm_pheno_metrics_avg = calculate_metrics_by_average(norm_pheno_metrics)

        denorm_pheno_metrics = calculate_phenotype_metrics_by_time(
            all_preds_denorm, all_targets_denorm, self.phenotype_names
        )
        denorm_pheno_metrics_avg = calculate_metrics_by_average(denorm_pheno_metrics)

        return {
            "normalized": {
                "phenotypes": norm_pheno_metrics,
                "overall": norm_pheno_metrics_avg,
            },
            "denormalized": {
                "phenotypes": denorm_pheno_metrics,
                "overall": denorm_pheno_metrics_avg,
            },
        }

    def _train_regular(self) -> dict[str, Any]:
        self.logger.info("=" * 80)
        self.logger.info("模拟数据常规训练")
        self.logger.info("=" * 80)

        self._log_configuration()

        start_time = time.time()

        # 构建数据加载器 (使用模拟数据专用接口, 已预划分)
        train_loader, val_loader, test_loader, scalers = get_sim_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            norm_method=self.norm_method,
            scaler_type=self.scaler_type,
            num_workers=self.num_workers,
            device=self.device,
        )

        save_scalers(scalers, self.scaler_file)
        self.logger.info(f"归一化器已保存至: {self.scaler_file}")
        self.logger.info("数据加载完成")

        optimizer, scheduler = self._create_optimizer_and_scheduler()
        criterion = self._create_loss()

        best_val_loss = float("inf")
        best_val_metrics = {}
        best_val_pheno_metrics = {phenotype: {} for phenotype in self.phenotype_names}
        best_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        patience_counter = 0
        best_epoch = -1

        train_loss_history = []
        val_loss_history = []
        train_metrics_history = []
        val_metrics_history = []
        val_pheno_metrics_history = []

        for epoch in range(self.epochs):
            avg_train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, criterion
            )
            avg_val_loss, val_pheno_metrics, val_metrics = self._validate_epoch(
                val_loader, criterion
            )

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            val_pheno_metrics_history.append(val_pheno_metrics)

            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Loss[train:{avg_train_loss:.6f} / val:{avg_val_loss:.6f}] | "
                f"PCC[train:{train_metrics['pcc']:.6f} / val:{val_metrics['pcc']:.6f}] | "
                f"R²[train:{train_metrics['r2']:.6f} / val:{val_metrics['r2']:.6f}] | "
                f"lr:{current_lr:.2e}"
                f"{' ★' if avg_val_loss < best_val_loss else ''}"
            )

            scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_pheno_metrics = val_pheno_metrics
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.info(f"连续 {self.patience} 轮无改善，触发早停！")
                break

        training_time = time.time() - start_time
        final_lr = optimizer.param_groups[0]["lr"]

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

        # 测试集评估
        self.logger.info("=" * 80)
        self.logger.info("测试集评估")
        self.logger.info("=" * 80)

        self.model.load_state_dict(best_model_state)
        scalers = load_scalers(self.scaler_file)
        self.logger.info(f"Scaler对象已从 {self.scaler_file} 加载")

        test_results = self._evaluate_test_set(test_loader, scalers)

        self.logger.info("（反归一化）数据的测试集整体指标:")
        for metric, value in test_results["denormalized"]["overall"].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
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
    ):
        # 1. 模型权重
        model_path = self.best_model_save_dir / "best_model.pt"
        cpu_model_state = {k: v.cpu().clone() for k, v in best_model_state.items()}
        torch.save(cpu_model_state, model_path)
        self.logger.info(f"【模型权重】已保存至: {model_path}")

        # 2. 训练摘要
        summary_info = {
            "exp_name": self.exp_name,
            "exp_group": self.exp_group,
            "rep_name": self.rep_name,
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

        def default_dump(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                summary_info, f, ensure_ascii=False, indent=4, default=default_dump
            )
        self.logger.info(f"【训练摘要】已保存至: {summary_path}")

        # 3. 训练历史
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

        # 4. 各表型各指标历史
        metric_names = ["mse", "rmse", "mae", "pcc", "r2"]
        for metric in metric_names:
            data = {"epoch": list(range(1, len(val_pheno_metrics_history) + 1))}
            for phenotype in self.phenotype_names:
                phenotype_data = []
                for epoch_metrics in val_pheno_metrics_history:
                    if (
                        phenotype in epoch_metrics
                        and metric in epoch_metrics[phenotype]
                    ):
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

    def train(self) -> dict[str, Any]:
        """模拟数据训练入口 (只支持常规训练)"""
        return self._train_regular()

    def _log_configuration(self) -> None:
        self.logger.info("训练配置:")
        self.logger.info(f"    实验名称: {self.exp_name}")
        self.logger.info(f"    实验组: {self.exp_group}")
        self.logger.info(f"    Rep: {self.rep_name}")
        self.logger.info(f"    样本数量: {self.num_sample}")
        self.logger.info(f"    特征数量: {self.snp_dim}")
        self.logger.info(
            f"    表型数量: {len(self.phenotype_names)} {self.phenotype_names}"
        )
        self.logger.info(f"    时间点数: {self.time_coordinates}")
        self.logger.info("    训练模式: 模拟数据常规训练 (预划分)")
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
        self.logger.info(f"    loss_type: {self.loss_type}")
        if self.loss_type == "mse_offset":
            self.logger.info(f"    offset_weight: {self.offset_weight}")
            self.logger.info(f"    pcc_weight: {self.pcc_weight}")
        elif self.loss_type == "mse_pcc":
            self.logger.info(f"    pcc_weight: {self.pcc_weight}")
        self.logger.info("模型配置:")
        self.logger.info(f"    snp_dim: {self.snp_dim}")
        self.logger.info(f"    d_model: {self.d_model}")
        self.logger.info(f"    num_heads: {self.num_heads}")
        self.logger.info(f"    num_layers: {self.num_layers}")
        self.logger.info(f"    num_embeddings: {self.num_embeddings}")
        self.logger.info(f"    d_ff: {self.d_ff}")
        self.logger.info(f"    degree: {self.degree}")
        self.logger.info(f"    dropout: {self.dropout}")
        self.logger.info("数据文件:")
        self.logger.info(f"    train_snp_csv: {self.config.train_snp_csv}")
        self.logger.info(f"    train_pheno_csv: {self.config.train_pheno_csv}")
        self.logger.info(f"    val_snp_csv: {self.config.val_snp_csv}")
        self.logger.info(f"    val_pheno_csv: {self.config.val_pheno_csv}")
        self.logger.info(f"    test_snp_csv: {self.config.test_snp_csv}")
        self.logger.info(f"    test_pheno_csv: {self.config.test_pheno_csv}")
        self.logger.info("路径配置:")
        self.logger.info(f"    model_save_dir: {self.best_model_save_dir}")
        self.logger.info(
            f"    training_history_save_dir: {self.training_history_save_dir}"
        )
        self.logger.info(f"    phenotype_scaler_dir: {self.scaler_save_dir}")
