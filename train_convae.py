"""ConvAE 数据准备/训练入口；--prepare_only 不初始化模型或导出潜在特征。"""

import argparse

from config import ConvAEConfig
from trainer import ConvAETrainer


def _str_to_bool(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("请显式使用 true 或 false")
    return value.lower() == "true"


def train_one():
    """沿用 ConvAE 名称；本模型是带表型辅助任务的 masked autoencoder。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--output_root", default=ConvAEConfig.model_fields["output_root"].default)
    parser.add_argument("--split_file", default=None, help="已有 TFPP 普通/fold manifest；不重新划分")
    parser.add_argument("--snp_file", default=ConvAEConfig.model_fields["snp_file"].default)
    parser.add_argument("--pheno_dir", default=ConvAEConfig.model_fields["pheno_dir"].default)
    parser.add_argument("--pheno_names", nargs="+", default=["RGBVI"])
    parser.add_argument("--time_coordinates", type=float, nargs="+")
    parser.add_argument("--chunk_size", type=int, default=12000)
    parser.add_argument("--overlap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare_only", action="store_true", help="仅保存数据管线，不初始化模型或训练")
    parser.add_argument("--use_pheno_help", type=_str_to_bool, default=True)
    parser.add_argument("--recon_weight", type=float, default=None)
    parser.add_argument("--pheno_weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_warmup", type=_str_to_bool, default=True)
    for name in ("patch_size", "d_model", "d_ext", "num_heads", "encoder_layers", "decoder_layers"):
        parser.add_argument(f"--{name}", type=int, default=ConvAEConfig.model_fields[name].default)
    for name in ("train_mask_ratio", "val_mask_ratio", "dropout"):
        parser.add_argument(f"--{name}", type=float, default=ConvAEConfig.model_fields[name].default)
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in
               ("split_file", "snp_file", "pheno_dir", "chunk_size", "overlap", "seed", "time_coordinates", "output_root")}
    if args.exp_name is not None:
        options["exp_name"] = args.exp_name
    if args.device is not None:
        options["device"] = args.device
    options.update({key: getattr(args, key) for key in
                    ("use_pheno_help", "use_warmup", "patch_size", "d_model", "d_ext", "num_heads",
                     "encoder_layers", "decoder_layers", "train_mask_ratio", "val_mask_ratio", "dropout")})
    config = ConvAEConfig(
        epochs=args.epochs,
        lr=0.0001,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recon_weight=args.recon_weight if args.recon_weight is not None else (0.7 if args.use_pheno_help else 1.0),
        pheno_weight=args.pheno_weight if args.pheno_weight is not None else (0.3 if args.use_pheno_help else 0.0),
        phenotype_names=args.pheno_names,
        **options,
    )

    trainer = ConvAETrainer(config=config, initialize_model=not args.prepare_only)

    # 开始训练
    if args.prepare_only:
        print(f"ConvAE 数据管线已保存至 {trainer.config.output_dir}")
    else:
        trainer.train()


if __name__ == "__main__":
    train_one()
