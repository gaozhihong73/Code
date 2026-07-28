import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

from config.ConvAE_Config import ConvAEConfig
from dataloader import ConvAEDataset, get_mae_dataloader
from model.ConvAE import ConvAE
from trainer._scheduler import WarmupSequentialLR
from utils import Logger


class ConvAETrainer:
    """ConvAE 训练器，集成 TFPP 风格的日志保存、Curriculum Learning (课程学习) 与指标追踪。"""

    def __init__(self, config: ConvAEConfig):
        """初始化训练器并构建数据集与模型。"""
        # 保存配置
        self.config = config

        # 保存目录参数
        self.pheno_dir = config.pheno_dir
        self.snp_file = config.snp_file

        self.best_model_save_dir = Path(config.best_model_save_dir)
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)

        self.training_history_save_dir = Path(config.training_history_save_dir)
        self.training_history_save_dir.mkdir(parents=True, exist_ok=True)

        # 数据集初始化
        self.phenotype_names = config.phenotype_names
        self.dataset = self._create_dataset()
        self.num_sample = len(self.dataset)
        self.snp_dim: int = self.dataset.snp.shape[1]
        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio

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

        # 模型初始化
        self.model = self._create_model()

        # ConvAE 专有
        self.use_pheno_help = config.use_pheno_help  # TODO 先不考虑不使用表型辅助的情况
        self.compressed_snp_save_dir = Path(config.compressed_snp_save_dir)
        self.compressed_snp_save_dir.mkdir(parents=True, exist_ok=True)

        # 损失函数设置：
        # SNPs 是 0, 1, 2, 3 的离散类别，使用交叉熵计算重建损失
        self.recon_criterion = nn.CrossEntropyLoss()
        # 表型属于连续变量，使用 MSE 损失
        self.pheno_criterion = nn.MSELoss()

        # 损失权重
        self.recon_weight = config.recon_weight
        self.pheno_weight = config.pheno_weight

        self.is_cuda = "cuda" in str(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.is_cuda)

    def _create_model(self) -> ConvAE:
        """获取模型。"""
        # 基于 DataLoader 计算出的切块数量，初始化 ConvAE
        return ConvAE(num_chunks=self.dataset.num_chunks).to(self.device)

    def _create_dataset(self) -> ConvAEDataset:
        """获取数据集对象。"""
        return ConvAEDataset(
            snp_file=self.snp_file,
            pheno_dir=self.pheno_dir,
            phenotype_names=self.phenotype_names,
        )

    def _create_optimizer_and_scheduler(self):
        """
        创建优化器：实现分层学习率 + 分层权重衰减
        """
        # 1. 创建优化器：直接传入模型的所有参数，使用统一的学习率和权重衰减
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.weight_decay
        )

        if self.config.use_warmup:
            # 获取每次训练的总批次数，用于支持 Step 级别的平滑更新
            # 如果你依然想保持 Epoch 级别更新，只需让：
            # total_steps = self.epochs，warmup_steps = self.warmup_epochs 即可
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=self.epochs - self.warmup_epochs, eta_min=self.min_lr
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

    # ==========================================================================
    # 核心循环：单轮训练与验证
    # ==========================================================================
    def _train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer
    ) -> tuple[float, float, float]:  # 传入 scheduler 以便按 step 更新
        """训练一个 epoch 并返回各项损失。

        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器。
        optimizer : optim.Optimizer
            优化器。

        Returns
        -------
        tuple[float, float, float]
            平均总损失、平均重建损失和平均表型损失。
        """
        self.model.train()
        num_batches = len(train_loader)
        metrics = {"total": 0.0, "recon": 0.0, "pheno": 0.0}

        optimizer.zero_grad()  # 在循环外初始化清空梯度

        for batch_idx, (X_chunks, pheno_target) in enumerate(train_loader):
            X_chunks = X_chunks.to(self.device)
            pheno_target = pheno_target.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.is_cuda):
                _, pred_phenotype, pred_snps, _ = self.model(X_chunks, mask_ratio=0.5)

                # 1. 计算重建损失
                if self.recon_weight > 0 and pred_snps is not None:
                    B, Num_Chunks, L = X_chunks.shape
                    target_snps = X_chunks.view(
                        B * Num_Chunks, self.model.num_patches, self.model.patch_size
                    )
                    pred_snps_ce = pred_snps.permute(0, 3, 1, 2)
                    recon_loss = self.recon_criterion(pred_snps_ce, target_snps)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)

                # 2. 计算表型损失
                if self.pheno_weight > 0:
                    pred_phenotype = pred_phenotype.view(pheno_target.shape[0], -1)
                    pheno_target_flat = pheno_target.view(pheno_target.shape[0], -1)
                    pheno_loss = self.pheno_criterion(pred_phenotype, pheno_target_flat)
                else:
                    pheno_loss = torch.tensor(0.0, device=self.device)

                # 3. 加权求和并除以累积步数（关键：对 loss 平均，保证梯度量级与大 Batch 一致）
                total_loss = (
                    self.recon_weight * recon_loss + self.pheno_weight * pheno_loss
                )
                loss_scaled = total_loss / self.accumulation_steps

            # 反向传播放大的梯度
            self.scaler.scale(loss_scaled).backward()

            # 触发权重更新的条件：达到了指定的累积步数，或者是当前 Epoch 的最后一个 Batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (
                batch_idx + 1
            ) == num_batches:
                # 优化器步进前，必须先 unscale 才能做梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip_norm
                )

                # 优化器更新与 Scaler 更新
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()  # 显式清空梯度，为下一次累积做准备

            # 记录指标时，依然使用原始未缩放的 total_loss.item()
            metrics["total"] += total_loss.item()
            metrics["recon"] += recon_loss.item()
            metrics["pheno"] += pheno_loss.item()

        return (
            metrics["total"] / num_batches,
            metrics["recon"] / num_batches,
            metrics["pheno"] / num_batches,
        )

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """验证一个 epoch 并返回各项损失。

        Parameters
        ----------
        val_loader : DataLoader
            验证数据加载器。

        Returns
        -------
        tuple[float, float, float]
            平均总损失、平均重建损失和平均表型损失。
        """
        self.model.eval()
        num_batches = len(val_loader)
        metrics = {"total": 0.0, "recon": 0.0, "pheno": 0.0}

        with torch.no_grad():
            for X_chunks, pheno_target in val_loader:
                X_chunks = X_chunks.to(self.device)
                pheno_target = pheno_target.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.is_cuda):
                    # 注意：在 eval 模式下，ConvAE.py 中的 mask_ratio 将被忽略，pred_snps 返回 None
                    _, pred_phenotype, _, _ = self.model(X_chunks, mask_ratio=0.0)

                    # 验证时主要关注表型损失，因为解码器此时不生效 (遵循 ConvAE.py 逻辑)
                    recon_loss = torch.tensor(0.0, device=self.device)

                    pred_phenotype = pred_phenotype.view(pheno_target.shape[0], -1)
                    pheno_target_flat = pheno_target.view(pheno_target.shape[0], -1)
                    pheno_loss = self.pheno_criterion(pred_phenotype, pheno_target_flat)

                    # 验证集的 Total Loss 保持与训练一致的加权
                    total_loss = (
                        self.recon_weight * recon_loss + self.pheno_weight * pheno_loss
                    )

                metrics["total"] += total_loss.item()
                metrics["recon"] += recon_loss.item()
                metrics["pheno"] += pheno_loss.item()

        return (
            metrics["total"] / num_batches,
            metrics["recon"] / num_batches,
            metrics["pheno"] / num_batches,
        )

    # ==========================================================================
    # 主训练流程与文件保存 (TFPP 风格)
    # ==========================================================================
    def train(self):
        """ConvAE 主训练流程。"""
        self.logger.info("=" * 80)
        self.logger.info("开始 ConvAE 训练")
        self.logger.info("=" * 80)
        self._log_configuration()

        # 根据 DataLoader 文件获取划分好的 loader
        train_loader, val_loader = get_mae_dataloader(
            self.dataset,
            self.batch_size,
            val_ratio=self.val_ratio,
            train_ratio=self.train_ratio,
            seed=self.seed,
            num_workers=self.num_workers,
        )

        # 创建优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler()

        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = -1
        patience_counter = 0

        # 用于存储日志的历史列表
        history = {
            "epoch": [],
            "train_total": [],
            "train_recon": [],
            "train_pheno": [],
            "val_total": [],
            "val_pheno": [],
            "lr": [],
        }

        start_time = time.time()

        for epoch in range(self.epochs):
            t_total, t_recon, t_pheno = self._train_epoch(
                train_loader,
                optimizer,
            )
            v_total, v_recon, v_pheno = self._validate_epoch(val_loader)

            # 3. 日志追踪
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch + 1)
            history["train_total"].append(t_total)
            history["train_recon"].append(t_recon)
            history["train_pheno"].append(t_pheno)
            history["val_total"].append(v_total)
            history["val_pheno"].append(v_pheno)
            history["lr"].append(current_lr)

            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Train: [rec:{t_recon:.6f} / phe:{t_pheno:.6f}] | "
                f"Val: [total:{v_total:.6f} / phe:{v_pheno:.6f}] | "
                f"LR: {current_lr:.2e} {'★' if v_total < best_val_loss else ''}"
            )

            scheduler.step()

            # 4. 早停与最优模型选取（Warmup 期间由于不看重表型，不计入早停判断）
            if v_total < best_val_loss:
                best_val_loss = v_total
                best_epoch = epoch + 1
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                self.logger.info(f"早停触发: {self.config.patience} 轮无改善")
                break

        total_time = time.time() - start_time
        self.logger.info(f"\n训练完成. 耗时 {total_time / 60:.2f} 分钟.")
        self.logger.info(
            f"最佳 Epoch: {best_epoch}, 最佳 Val Loss: {best_val_loss:.6f}"
        )

        # 5. 保存权重、日志及实施特征压缩
        if self.config.is_save_training_info and best_model_state is not None:
            self._save_model_info(best_model_state, best_val_loss, best_epoch, history)
            self._compress_data(best_model_state)

    def _save_model_info(
        self, state: dict[str, Any], best_loss: float, best_epoch: int, history: dict
    ):
        """
        TFPP 风格的模型与元数据存储
        """
        # 1. 保存模型参数
        model_path = self.best_model_save_dir / "best_model.pt"
        torch.save(state, model_path)
        self.logger.info(f"【模型权重】已保存至: {model_path}")

        # 2. 保存 JSON 摘要
        summary_info = {
            "best_epoch": int(best_epoch),
            "total_epochs": int(self.epochs),
            "best_val_loss": float(best_loss),
            "warmup_epochs": int(self.warmup_epochs),
        }
        summary_path = self.training_history_save_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=4)

        # 3. 保存 CSV 训练历史曲线
        df = pd.DataFrame(history)
        history_path = self.training_history_save_dir / "training_history.csv"
        df.to_csv(history_path, index=False)
        self.logger.info(
            f"【训练历史及摘要】已保存至: {self.training_history_save_dir}"
        )

    def _compress_data(self, best_model_state: dict[str, Any]):
        """
        全量数据压缩逻辑：提取编码器最终的隐空间特征并落盘
        """
        self.logger.info("=" * 80)
        self.logger.info("开始全量数据特征压缩...")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.model.eval()

        # 复用 dataset 提取纯数据
        compress_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        compressed_features = []

        with torch.no_grad():
            for X_chunks, _ in compress_loader:
                X_chunks = X_chunks.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.is_cuda):
                    # 调用模型本身前向获得压缩特征，此时掩码比率一定为 0
                    compressed_final_X, _, _, _ = self.model(X_chunks, mask_ratio=0.0)

                    # 展平特征：[B, Num_Chunks, d_ext] -> [B, Num_Chunks * d_ext]
                    batch_features = compressed_final_X.view(X_chunks.shape[0], -1)
                    compressed_features.append(batch_features.cpu().numpy())

        final_array = np.concatenate(compressed_features, axis=0)
        save_path = self.compressed_snp_save_dir / "snp_compressed.npy"
        np.save(str(save_path), final_array)

        self.logger.info(f"压缩结果已保存: {save_path} (Shape: {final_array.shape})")

    def _log_configuration(self):
        """记录配置信息"""
        self.logger.info(f"样本数量: {self.num_sample}")
        self.logger.info(f"特征数量: {self.snp_dim}")
        self.logger.info(
            f"表型数量: {len(self.phenotype_names)} {self.phenotype_names}"
        )
        self.logger.info(f"use_pheno_help: {self.use_pheno_help}")
        self.logger.info(f"train_ratio: {self.train_ratio}")
        self.logger.info(f"val_ratio: {self.val_ratio}")
        self.logger.info(f"recon_weight: {self.recon_weight}")
        self.logger.info(f"pheno_weight: {self.pheno_weight}")
        self.logger.info(f"epochs: {self.epochs}")
        self.logger.info(f"lr: {self.lr}")
        self.logger.info(f"weight_decay: {self.weight_decay}")
        self.logger.info(f"batch_size: {self.batch_size * self.accumulation_steps}")
        self.logger.info(f"patience: {self.patience}")
        self.logger.info(f"grad_clip_norm: {self.grad_clip_norm}")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"use_warmup: {self.use_warmup}")
        self.logger.info(f"snp_file: {self.snp_file}")
        self.logger.info(f"pheno_dir: {self.pheno_dir}")
        self.logger.info(f"model_save_dir: {self.best_model_save_dir}")
        self.logger.info(f"training_history_save_dir: {self.training_history_save_dir}")
        self.logger.info(f"compressed_snp_save_dir: {self.compressed_snp_save_dir}")
