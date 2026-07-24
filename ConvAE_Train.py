from config import ConvAEConfig
from trainer import ConvAETrainer


def train_one():
    """
    训练单个实验组合
    """
    config = ConvAEConfig(
        epochs=100,
        lr=0.0001,
        weight_decay=1e-4,
        batch_size=4,
        num_workers=8,
        recon_weight=0.7,
        pheno_weight=0.3,
    )

    trainer = ConvAETrainer(config=config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    train_one()
