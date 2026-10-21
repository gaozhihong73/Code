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
from dataloader import ConvAEDataset, get_mae_dataloaders
from model.ConvAE import ConvAE, MASK_POLICY, MODEL_VERSION
from trainer._scheduler import WarmupSequentialLR
from utils import Logger
from utils.Inference import file_sha256
from utils.ConvAE_Objective import (
    EMPTY_TARGET_POLICY, RECONSTRUCTION_POLICY, convae_loss_terms, fixed_reconstruction_masks,
)
from utils.Split import (IndividualSplit, default_split_path, load_or_create_split, load_split)
from utils.ConvAE_Preprocessing import convae_split_fingerprint, save_convae_preprocessing


class ConvAETrainer:
    """ConvAE 训练器"""

    def __init__(self, config: ConvAEConfig, *, indices: IndividualSplit | None = None,
                 initialize_model: bool = True):
        """初始化训练器并构建数据集与模型。"""
        # 保存配置
        self.config = config

        # 保存目录参数
        self.pheno_dir = config.pheno_dir
        self.snp_file = config.snp_file

        # 数据集初始化
        self.phenotype_names = config.phenotype_names
        self.dataset = self._create_dataset()
        self.num_sample = len(self.dataset)
        self.snp_dim: int = self.dataset.snp.shape[1]
        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio

        # 固定共享划分、只在 train 拟合 scaler，并在创建模型前持久化数据契约。
        self._prepare_data(indices)
        config = self.config
        self.best_model_save_dir = Path(config.best_model_save_dir)
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
        self.training_history_save_dir = Path(config.training_history_save_dir)
        self.training_history_save_dir.mkdir(parents=True, exist_ok=True)

        # 日志记录器
        self.log_file = config.log_file
        self.logger = Logger(log_filename=self.log_file, unique=False)

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
        torch.manual_seed(config.seed)
        self.model = self._create_model() if initialize_model else None
        self.train_mask_generator = torch.Generator(device=config.device).manual_seed(config.seed)
        self.selection_metric = config.selection_metric
        self._prepare_validation_masks()

        # ConvAE 专有
        self.use_pheno_help = config.use_pheno_help
        self.compressed_snp_save_dir = Path(config.compressed_snp_save_dir)

        # 损失函数设置：
        # SNPs 是 0, 1, 2, 3 的离散类别，使用交叉熵计算重建损失
        self.recon_criterion = nn.CrossEntropyLoss()
        # 表型属于连续变量，使用 MSE 损失
        self.pheno_criterion = nn.MSELoss()

        # 损失权重
        self.recon_weight = config.recon_weight
        self.pheno_weight = config.pheno_weight

        self.is_cuda = "cuda" in str(self.device)
        self.scaler = torch.GradScaler(device="cuda", enabled=self.is_cuda)

    def _prepare_data(self, indices: IndividualSplit | None = None) -> None:
        """接受 TFPP 普通/fold manifest 或内存 IndividualSplit，不猜测 fold 划分。"""
        config = self.config
        if config.n_time_points is not None and config.n_time_points != self.dataset.time_points:
            raise ValueError("phenotype 列数、time_coordinates 与 n_time_points 必须一致")
        if config.n_phenotypes != self.dataset.phenotype.shape[1]:
            raise ValueError("phenotype 维数与 n_phenotypes 不一致")
        config = config.model_copy(update={"n_time_points": self.dataset.time_points})
        self.config = config
        source = None
        if indices is not None:
            if config.split_file is not None:
                raise ValueError("传入 IndividualSplit 时不要同时设置 split_file")
            split = indices
        elif config.split_file not in (None, "auto"):
            source = Path(config.split_file)
            if not source.is_file():
                raise FileNotFoundError(
                    f"共享 split/fold 文件不存在: {source}；请先保存 TFPP manifest，"
                    "或使用 split_file=None/auto 生成普通共享划分，禁止猜测 fold"
                )
            split = load_split(source, n_samples=self.num_sample)
        else:
            ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
            source = default_split_path(self.num_sample, ratios=ratios, seed=config.seed)
            split = load_or_create_split(source, self.num_sample, ratios=ratios, seed=config.seed)
        fingerprint = convae_split_fingerprint(split)
        data = get_mae_dataloaders(
            self.dataset, split, batch_size=config.batch_size,
            num_workers=config.num_workers, seed=config.seed, include_sample_ids=True,
        )
        fold = split.metadata.get("fold_index")
        tag = "split" if fold is None else f"fold_{fold:02d}"
        tag += "_" + fingerprint[:16]
        self.config = config.model_copy(update={"split_tag": tag})
        self.data_pipeline = data
        self.split, self.scalers = data.split, data.scalers
        self.train_loader, self.val_loader, self.test_loader = (
            data.train_loader, data.val_loader, data.test_loader,
        )
        self.preprocessing_metadata = save_convae_preprocessing(
            self.dataset, data, self.config, split_source=source,
        )

    def _create_model(self) -> ConvAE:
        """获取模型。"""
        config = self.config
        return ConvAE(
            chunk_size=self.dataset.chunk_size, patch_size=config.patch_size,
            num_chunks=self.dataset.num_chunks, d_model=config.d_model, d_ext=config.d_ext,
            n_phenotypes=config.n_phenotypes, n_time_points=config.n_time_points,
            use_pheno_help=config.use_pheno_help, num_heads=config.num_heads,
            encoder_layers=config.encoder_layers, decoder_layers=config.decoder_layers,
            dropout=config.dropout,
        ).to(self.device)

    def _prepare_validation_masks(self) -> None:
        """按 val sample_id 独立固定 mask，保存以便复现模型选择口径。"""
        ids = self.val_loader.dataset.sample_ids
        self.validation_masks = fixed_reconstruction_masks(
            self.dataset.chunk_padding_mask, patch_size=self.config.patch_size, sample_ids=ids,
            seed=self.config.seed, split_fingerprint=self.preprocessing_metadata["split_fingerprint"],
            mask_ratio=self.config.val_mask_ratio,
        )
        self.validation_mask_rows = {json.dumps(value.item() if isinstance(value, np.generic) else value): i
                                     for i, value in enumerate(ids)}
        path = Path(self.config.output_dir) / "preprocessing" / "validation_masks.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["sample_ids"], ids)
                        or not np.array_equal(saved["patch_masks"], self.validation_masks.numpy())):
                    raise ValueError("已有验证掩码与当前 seed/split/sample_id 不一致，请使用新实验目录")
        else:
            np.savez_compressed(path, sample_ids=ids, patch_masks=self.validation_masks.numpy())
        self.validation_masks_sha256 = file_sha256(path)

    def _validation_mask_for_ids(self, sample_ids) -> torch.Tensor:
        values = sample_ids.tolist() if isinstance(sample_ids, torch.Tensor) else list(sample_ids)
        try:
            rows = [self.validation_mask_rows[json.dumps(value)] for value in values]
        except KeyError as error:
            raise ValueError("验证 batch 中的 sample_id 不属于保存的 val 集合") from error
        return self.validation_masks[rows].to(self.device)

    def _create_dataset(self) -> ConvAEDataset:
        """获取数据集对象。"""
        return ConvAEDataset(
            snp_file=self.snp_file,
            pheno_dir=self.pheno_dir,
            phenotype_names=self.phenotype_names,
            chunk_size=self.config.chunk_size, overlap=self.config.overlap,
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
                optimizer, T_max=max(1, self.epochs - self.warmup_epochs), eta_min=self.min_lr
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
    def _run_epoch(self, loader: DataLoader, optimizer=None) -> tuple[float, float, float]:
        """训练/验证共用同一目标，按 SNP/表型监督元素数汇总，避免 batch 大小改变口径。"""
        training = optimizer is not None
        self.model.train(training)
        stats = dict(recon_sum=0.0, recon_count=0, pheno_sum=0.0, pheno_count=0, empty_batches=0)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pending_grad = False
        for batch_idx, (chunks, phenotype, sample_ids) in enumerate(loader):
            chunks, phenotype = chunks.to(self.device), phenotype.to(self.device)
            padding = self.dataset.chunk_padding_mask.to(self.device)
            selected = None if training else self._validation_mask_for_ids(sample_ids)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type="cuda", enabled=self.is_cuda):
                    outputs = self.model(
                        chunks, padding_mask=padding,
                        mask_ratio=self.config.train_mask_ratio if training else self.config.val_mask_ratio,
                        reconstruction_mask=selected, generator=self.train_mask_generator if training else None,
                    )
                    terms = convae_loss_terms(outputs, chunks, phenotype, padding,
                                              use_pheno_help=self.use_pheno_help)
                    loss = terms.total(self.recon_weight, self.pheno_weight)
                has_signal = ((terms.recon_count > 0 and self.recon_weight > 0)
                              or (terms.pheno_count > 0 and self.pheno_weight > 0))
                if training and has_signal:
                    window_start = (batch_idx // self.accumulation_steps) * self.accumulation_steps
                    window_size = min(self.accumulation_steps, len(loader) - window_start)
                    self.scaler.scale(loss / window_size).backward()
                    pending_grad = True
            if training and ((batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == len(loader)):
                if pending_grad:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_grad = False
            stats["recon_sum"] += float(terms.recon_sum.detach())
            stats["recon_count"] += terms.recon_count
            stats["pheno_sum"] += float(terms.pheno_sum.detach())
            stats["pheno_count"] += terms.pheno_count
            stats["empty_batches"] += int(terms.recon_count == 0)
        recon = stats["recon_sum"] / max(1, stats["recon_count"])
        pheno = stats["pheno_sum"] / max(1, stats["pheno_count"])
        setattr(self, "last_train_stats" if training else "last_val_stats", stats)
        return self.recon_weight * recon + self.pheno_weight * pheno, recon, pheno

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> tuple[float, float, float]:
        return self._run_epoch(train_loader, optimizer)

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float, float]:
        return self._run_epoch(val_loader)

    def train(self):
        """只用固定 val 掩码和预先指定的选择指标早停，test 不参与模型选择。"""
        if self.model is None:
            raise ValueError("initialize_model=False 仅准备数据管线，不能训练；请显式初始化模型")
        self._log_configuration()
        self.logger.info(f"模型选择固定为 {self.selection_metric}，mask_policy={MASK_POLICY}")
        optimizer, scheduler = self._create_optimizer_and_scheduler()
        best_loss, best_epoch, best_state, patience = float("inf"), -1, None, 0
        keys = ("epoch", "train_total_loss", "train_recon_loss", "train_pheno_loss", "val_total_loss",
                "val_recon_loss", "val_pheno_loss", "selection_metric", "selection_loss", "lr",
                "train_recon_targets", "val_recon_targets", "val_empty_recon_batches")
        history = {key: [] for key in keys}
        start = time.time()
        for epoch in range(self.epochs):
            train = self._train_epoch(self.train_loader, optimizer)
            val = self._validate_epoch(self.val_loader)
            selected = val[0] if self.use_pheno_help else val[1]
            values = (epoch + 1, *train, *val, self.selection_metric, selected,
                      optimizer.param_groups[0]["lr"], self.last_train_stats["recon_count"],
                      self.last_val_stats["recon_count"], self.last_val_stats["empty_batches"])
            for key, value in zip(keys, values):
                history[key].append(value)
            self.logger.info(
                f"[{epoch + 1}/{self.epochs}] train_total={train[0]:.6f} recon={train[1]:.6f} pheno={train[2]:.6f} | "
                f"val_total={val[0]:.6f} recon={val[1]:.6f} pheno={val[2]:.6f} | "
                f"{self.selection_metric}={selected:.6f}, recon_targets={self.last_val_stats['recon_count']}"
            )
            scheduler.step()
            if not np.isfinite(selected):
                raise ValueError("验证选择损失不是有限值，禁止保存或用 NaN 选择 checkpoint")
            if selected < best_loss:
                best_loss, best_epoch, patience = selected, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                patience += 1
            if patience >= self.patience:
                break
        if best_state is None:
            raise ValueError("没有有效的验证 checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        if self.is_save_training_info:
            self._save_model_info(best_state, best_loss, best_epoch, history)
        return {"best_epoch": best_epoch, "selection_metric": self.selection_metric,
                "best_selection_loss": best_loss, "history": history, "training_time": time.time() - start}

    def _save_model_info(self, state: dict[str, Any], best_loss: float, best_epoch: int, history: dict):
        """结构化 checkpoint 明确绑定预处理、固定验证 mask 和预先指定的选择目标。"""
        index = best_epoch - 1
        metadata = {
            "schema_version": 3, "model_version": MODEL_VERSION,
            "selection_metric": self.selection_metric, "use_pheno_help": self.use_pheno_help,
            "selection_definition": self.config.selection_definition,
            "reconstruction_loss_definition": self.config.reconstruction_loss_definition,
            "phenotype_loss_definition": self.config.phenotype_loss_definition,
            "recon_weight": self.recon_weight, "pheno_weight": self.pheno_weight,
            "best_epoch": best_epoch, "best_selection_loss": best_loss,
            "val_total_loss": history["val_total_loss"][index],
            "val_recon_loss": history["val_recon_loss"][index],
            "val_pheno_loss": history["val_pheno_loss"][index],
            "val_recon_targets": history["val_recon_targets"][index],
            "val_empty_recon_batches": history["val_empty_recon_batches"][index],
            "mask_policy": MASK_POLICY, "reconstruction_policy": RECONSTRUCTION_POLICY,
            "empty_target_policy": EMPTY_TARGET_POLICY,
            "empty_training_batch_policy": "skip_optimizer_step_if_no_weighted_target",
            "val_mask_ratio": self.config.val_mask_ratio, "train_mask_ratio": self.config.train_mask_ratio,
            "validation_masks_sha256": self.validation_masks_sha256,
            "preprocessing_sha256": file_sha256(Path(self.config.output_dir) / "preprocessing/preprocessing.json"),
        }
        model_config = self.model.model_config()
        torch.save({"format_version": 2, "model_config": model_config, "metadata": metadata,
                    "state_dict": state}, self.best_model_save_dir / "best_model.pt")
        (self.best_model_save_dir / "model_config.json").write_text(
            json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.best_model_save_dir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.training_history_save_dir / "training_summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(history).to_csv(self.training_history_save_dir / "training_history.csv", index=False)

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
                with torch.autocast("cuda", enabled=self.is_cuda):
                    # 调用模型本身前向获得压缩特征，此时掩码比率一定为 0
                    compressed_final_X, _, _, _ = self.model(X_chunks, mask_ratio=0.0, padding_mask=self.dataset.chunk_padding_mask.to(self.device))

                    # 展平特征：[B, Num_Chunks, d_ext] -> [B, Num_Chunks * d_ext]
                    batch_features = compressed_final_X.view(X_chunks.shape[0], -1)
                    compressed_features.append(batch_features.cpu().numpy())

        final_array = np.concatenate(compressed_features, axis=0)
        save_path = self.compressed_snp_save_dir / "snp_compressed.npy"
        np.save(str(save_path), final_array)

        self.logger.info(f"压缩结果已保存: {save_path} (Shape: {final_array.shape})")

    def _log_configuration(self):
        """记录配置信息"""
        self.logger.info("=" * 60)
        self.logger.info("ConvAE 训练配置:")
        self.logger.info(f"    样本数量: {self.num_sample}")
        self.logger.info(f"    特征数量 (snp_dim): {self.snp_dim}")
        self.logger.info(
            f"    表型数量: {len(self.phenotype_names)} {self.phenotype_names}"
        )
        self.logger.info(f"    use_pheno_help: {self.use_pheno_help}")
        self.logger.info(f"    n_phenotypes/n_time_points: {self.config.n_phenotypes}/{self.config.n_time_points}")
        self.logger.info(f"    selection_metric: {self.selection_metric}")
        self.logger.info(f"    train_ratio: {self.train_ratio}")
        self.logger.info(f"    val_ratio: {self.val_ratio}")
        self.logger.info(f"    test_ratio: {self.test_ratio}")
        self.logger.info(f"    split_tag: {self.config.split_tag}")
        self.logger.info(f"    split_sha256: {self.preprocessing_metadata['split_sha256']}")
        self.logger.info(f"    train/val/test: {len(self.split.train_idx)}/{len(self.split.val_idx)}/{len(self.split.test_idx)}")

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
        self.logger.info("损失函数:")
        self.logger.info(f"    recon_weight: {self.recon_weight}")
        self.logger.info(f"    pheno_weight: {self.pheno_weight}")
        self.logger.info("    recon_criterion: CrossEntropyLoss")
        self.logger.info("    pheno_criterion: MSELoss")

        self.logger.info("-" * 60)
        self.logger.info("路径配置:")
        self.logger.info(f"    snp_file: {self.snp_file}")
        self.logger.info(f"    pheno_dir: {self.pheno_dir}")
        self.logger.info(f"    model_save_dir: {self.best_model_save_dir}")
        self.logger.info(
            f"    training_history_save_dir: {self.training_history_save_dir}"
        )
        self.logger.info(f"    compressed_snp_save_dir: {self.compressed_snp_save_dir}")
        self.logger.info(f"    log_file: {self.log_file}")
        self.logger.info("=" * 60)
