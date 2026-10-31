"""四个逐时间点深度学习比较模型的统一训练适配器。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .adapters import BaseAdapter, FitResult
from .deep_models import (
    CropformerModel,
    DLGWASModel,
    DNNGPSNPModel,
    WheatGPModel,
)


class SingleTimePointDeepAdapter(BaseAdapter):
    """每个时间点独立建模，validation 早停，最终拼接轨迹。"""

    def __init__(self, *, name: str, seed: int, options: dict[str, Any]):
        super().__init__(seed=seed, options=options)
        self.name = name
        requested = str(options.get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"{name} 请求 CUDA，但当前环境不可用")
        self.device = torch.device(requested)
        self.batch_size = int(options.get("batch_size", 32))
        self.max_epochs = int(options.get("max_epochs", 100))
        self.patience = int(options.get("patience", 10))
        self.learning_rate = float(options.get("learning_rate", 1e-3))
        self.weight_decay = float(options.get("weight_decay", 1e-4))
        self.min_delta = float(options.get("min_delta", 1e-6))
        self.grad_clip_norm = float(options.get("grad_clip_norm", 1.0))
        if min(self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("batch_size、max_epochs 和 patience 必须为正数")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate 必须为正且 weight_decay 不得为负")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _set_seed(self, value: int) -> None:
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

    def _fit_input_transform(self, values: np.ndarray) -> None:
        if self.name in ("DNNGP-SNP", "Cropformer"):
            self.input_mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
            self.input_scale = np.where(scale > 1e-12, scale, 1.0).astype(np.float32)

    def _transform_input(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if self.name == "DLGWAS":
            if not np.isin(values, (0, 1, 2)).all():
                raise ValueError("DLGWAS 只接受填补后的 0/1/2 剂量")
            encoded = np.eye(4, dtype=np.float32)[values.astype(np.int64) + 1]
            return encoded.transpose(0, 2, 1)
        if self.name == "WheatGP":
            return (values.astype(np.float32) / 2.0)[:, None, :]
        standardized = (
            values.astype(np.float32) - self.input_mean[None, :]
        ) / self.input_scale[None, :]
        return standardized[:, None, :]

    def _build_model(self, input_shape: tuple[int, ...]) -> nn.Module:
        if self.name == "DNNGP-SNP":
            return DNNGPSNPModel(
                channels=tuple(self.options.get("channels", [8, 16, 32])),
                pooled_length=int(self.options.get("pooled_length", 16)),
                dense_units=int(self.options.get("dense_units", 64)),
                dropout_input=float(self.options.get("dropout_input", 0.25)),
                dropout_dense=float(self.options.get("dropout_dense", 0.25)),
            )
        if self.name == "DLGWAS":
            return DLGWASModel(
                channels=int(self.options.get("channels", 16)),
                pooled_length=int(self.options.get("pooled_length", 16)),
                dropout=float(self.options.get("dropout", 0.75)),
            )
        if self.name == "WheatGP":
            if input_shape[-1] < 5:
                raise ValueError("WheatGP 的五段输入要求至少5个 SNP")
            return WheatGPModel(
                segment_count=5,
                segment_tokens=int(self.options.get("segment_tokens", 16)),
                hidden_size=int(self.options.get("hidden_size", 128)),
                cnn_dropout=float(self.options.get("cnn_dropout", 0.5)),
                lstm_dropout=float(self.options.get("lstm_dropout", 0.3)),
            )
        if self.name == "Cropformer":
            return CropformerModel(
                input_snps=input_shape[-1],
                embedding_dim=int(self.options.get("embedding_dim", 32)),
                num_heads=int(self.options.get("num_heads", 4)),
                max_tokens=int(self.options.get("max_tokens", 128)),
                hidden_dropout=float(self.options.get("hidden_dropout", 0.5)),
                attention_dropout=float(self.options.get("attention_dropout", 0.5)),
            )
        raise ValueError(f"未知深度模型: {self.name}")

    def _loader(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(features, target),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def _epoch_loss(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for features, target in loader:
                features = features.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                prediction = model(features)
                loss_sum += float(torch.sum((prediction - target).square()).item())
                count += len(target)
        return loss_sum / count

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        self._fit_input_transform(x_train)
        train_features_np = self._transform_input(x_train)
        val_features_np = self._transform_input(x_val)
        train_features = torch.from_numpy(train_features_np)
        val_features = torch.from_numpy(val_features_np)
        self.input_shape = tuple(train_features.shape[1:])
        self.target_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        target_scale = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
        self.target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
        train_target = (y_train - self.target_mean) / self.target_scale
        val_target = (y_val - self.target_mean) / self.target_scale

        self.models: list[nn.Module] = []
        history: list[dict[str, Any]] = []
        best_epochs: list[int] = []
        total_parameters = 0
        for time_index in range(y_train.shape[1]):
            model_seed = self.seed + time_index
            self._set_seed(model_seed)
            model = self._build_model(self.input_shape).to(self.device)
            total_parameters += sum(parameter.numel() for parameter in model.parameters())
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            train_loader = self._loader(
                train_features,
                torch.from_numpy(train_target[:, time_index].astype(np.float32)),
                shuffle=True,
                seed=model_seed,
            )
            val_loader = self._loader(
                val_features,
                torch.from_numpy(val_target[:, time_index].astype(np.float32)),
                shuffle=False,
                seed=model_seed,
            )
            best_loss = float("inf")
            best_state = None
            best_epoch = 0
            stale = 0
            for epoch in range(1, self.max_epochs + 1):
                model.train()
                train_loss_sum = 0.0
                train_count = 0
                for features, target in train_loader:
                    features = features.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(features)
                    loss = nn.functional.mse_loss(prediction, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                    optimizer.step()
                    train_loss_sum += float(loss.item()) * len(target)
                    train_count += len(target)
                val_loss = self._epoch_loss(model, val_loader)
                history.append(
                    {
                        "time_index": time_index,
                        "model_seed": model_seed,
                        "epoch": epoch,
                        "train_mse_normalized": train_loss_sum / train_count,
                        "val_mse_normalized": val_loss,
                        "val_rmse": float(np.sqrt(val_loss) * self.target_scale[time_index]),
                    }
                )
                if val_loss < best_loss - self.min_delta:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            if best_state is None:
                raise RuntimeError(f"{self.name} time_index={time_index} 未产生有效 checkpoint")
            model.load_state_dict(best_state)
            self.models.append(model.cpu())
            best_epochs.append(best_epoch)

        parameters = {
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": str(self.device),
            "model_seed_formula": "seed + zero_based_time_index",
            "timepoint_model_count": len(self.models),
        }
        return FitResult(
            best_parameters=parameters,
            best_epoch=max(best_epochs),
            validation_history=history,
            effective_feature_count=int(x_train.shape[1]),
            parameter_count=int(total_parameters),
            best_epochs_by_time=best_epochs,
        )

    def predict(self, x_test) -> np.ndarray:
        features = torch.from_numpy(self._transform_input(x_test))
        prediction = np.empty((len(features), len(self.models)), dtype=np.float32)
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        for time_index, model in enumerate(self.models):
            model = model.to(self.device)
            blocks = []
            model.eval()
            with torch.no_grad():
                for (batch,) in loader:
                    blocks.append(model(batch.to(self.device, non_blocking=True)).cpu())
            normalized = torch.cat(blocks).numpy()
            prediction[:, time_index] = (
                normalized * self.target_scale[time_index] + self.target_mean[time_index]
            )
            self.models[time_index] = model.cpu()
        return prediction

    def preprocessing_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "target_mean_by_time": self.target_mean,
            "target_scale_by_time": self.target_scale,
        }
        if hasattr(self, "input_mean"):
            arrays.update(
                input_marker_mean=self.input_mean,
                input_marker_scale=self.input_scale,
            )
        return arrays

    def save_artifacts(self, output_dir: Path) -> dict[str, Any]:
        path = output_dir / "timepoint_model_states.pt"
        torch.save(
            {
                "format_version": "e3_timepoint_state_dicts_v1",
                "model_name": self.name,
                "input_shape": list(self.input_shape),
                "state_dicts": [model.state_dict() for model in self.models],
            },
            path,
        )
        return {
            "checkpoint_path": str(path),
            "checkpoint_format": "torch_state_dicts_weights_only_compatible_v1",
        }

    def resource_metrics(self) -> dict[str, float | None]:
        if self.device.type != "cuda":
            return {"peak_gpu_memory_mb": None}
        return {
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(self.device))
            / (1024.0 * 1024.0)
        }
