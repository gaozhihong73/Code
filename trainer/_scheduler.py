"""轻量级 WarmupSequentialLR

用途: 替代 PyTorch 的 SequentialLR 用于 [LinearLR warmup -> CosineAnnealingLR main] 两段调度.

背景:
    PyTorch 官方 SequentialLR 在跨过里程碑时, 会显式调用 sub-scheduler.step(epoch=0),
    以便让主调度器从 last_epoch=0 重新计数. 这个 step(epoch=0) 会触发
    `EPOCH_DEPRECATION_WARNING` (见 torch.optim.lr_scheduler._LRScheduler.step 的 epoch 参数分支).
    本 wrapper 用闭式公式直接算 lr, 不再走那条 deprecate 路径, 但 lr 曲线与
    SequentialLR(LinearLR(start_factor=s, end_factor=1.0, total_iters=w),
                 CosineAnnealingLR(T_max=T - w, eta_min=eta))
    完全等价.

约束 (与三个 trainer 当前调用一致即可):
    - LinearLR end_factor 固定为 1.0 (现行所有调用都是 1.0)
    - 仅有两段: warmup -> cosine, 单个 milestone
"""

import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupSequentialLR(_LRScheduler):
    """[LinearLR warmup] -> [CosineAnnealingLR main] 两段 LR 调度器

    与 SequentialLR 接口兼容:
        WarmupSequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        """初始化调度器并预设初始学习率。"""
        if len(schedulers) != 2 or len(milestones) != 1:
            raise ValueError(
                "WarmupSequentialLR 当前仅支持 [LinearLR -> CosineAnnealingLR] 两段 + "
                "单个 milestone 的配置"
            )
        self._warmup = schedulers[
            0
        ]  # LinearLR 子对象 (只用其元数据: start_factor/end_factor/total_iters/base_lrs)
        self._main = schedulers[
            1
        ]  # CosineAnnealingLR 子对象 (用 eta_min/T_max/base_lrs)
        self._warmup_epochs = milestones[0]
        self.optimizer = optimizer
        # 与 PyTorch _LRScheduler 约定一致: last_epoch 起点设为 0, 并把 lr 预设到第 1 个训练
        # epoch 用到的值 (= warmup 起点); 之后每次 step() 把 last_epoch + 1, 计算下一轮 lr.
        self.last_epoch = last_epoch + 1
        self._step_count = 0
        init_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, init_lrs, strict=True):
            group["lr"] = lr
        self._last_lr = init_lrs

    # === 与 _LRScheduler 接口保持一致 ===

    @property
    def base_lrs(self) -> list[float]:  # type: ignore[override]
        """返回各参数组的基础学习率列表。"""
        return self._warmup.base_lrs

    def get_last_lr(self):
        """返回上一次 step 后的学习率列表。"""
        return self._last_lr

    def get_lr(self):
        """根据 self.last_epoch 计算 lr 列表

        - last_epoch < warmup_epochs: 走 LinearLR 闭式, lr = base_lr * (start + (1-start)*min(w,last)/w)
        - last_epoch == warmup_epochs: 起点 lr = base_lr (LinearLR end_factor=1.0)
        - last_epoch >  warmup_epochs: 走 CosineAnnealingLR 闭式,
                  lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * local / T_max)) / 2
        """
        if self.last_epoch < self._warmup_epochs:
            # warmup 阶段 (LinearLR 闭式, last_epoch 从 0 计起)
            s = self._warmup.start_factor
            w = self._warmup.total_iters
            end = self._warmup.end_factor  # 当前所有调用都是 1.0, 但保留式
            factor = s + (end - s) * min(w, self.last_epoch) / w
            return [base_lr * factor for base_lr in self._warmup.base_lrs]

        # cosine 阶段: 距 milestone 的偏移
        local = self.last_epoch - self._warmup_epochs  # 0 起点
        if local == 0:
            # 跨过 milestone: lr 等于 warmup 终点 = base_lr * end_factor
            return [
                base_lr * self._warmup.end_factor for base_lr in self._main.base_lrs
            ]

        T_max = self._main.T_max
        eta_min = self._main.eta_min
        return [
            eta_min + (base_lr - eta_min) * (1 + math.cos(local * math.pi / T_max)) / 2
            for base_lr in self._main.base_lrs
        ]

    def step(self, epoch=None):
        """前进一步 lr

        重要: 不接受 epoch 参数 (PyTorch 上游 deprecate 该路径, 这里拒绝以彻底消除 warning).
        """
        if epoch is not None:
            raise RuntimeError(
                "WarmupSequentialLR.step() 不接受 epoch 参数 (PyTorch 已 deprecate 该路径)"
            )
        self._step_count += 1
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs, strict=True):
            group["lr"] = lr
        self._last_lr = new_lrs
