# E3 deep-model provenance

The E3 adapters use the same frozen individual split and do not import upstream
data loaders, feature selectors, validation splits, pretrained weights, or test
logic.

## DNNGP-SNP

- Paper: Wang et al., *Molecular Plant* (2023), DOI
  `10.1016/j.molp.2022.11.004`.
- Implementation: independently written from the published architecture. No
  source code from the unlicensed official repository is copied.
- Adaptation: SNP-only input, three convolution layers, one BatchNorm layer,
  two dropout layers, adaptive pooling, dense regression head, and one model
  per time point.

## DLGWAS

- Paper: Liu et al., *Frontiers in Genetics* (2019), DOI
  `10.3389/fgene.2019.01091`.
- Implementation: independently written from the published dual-stream
  architecture. No source code from the unlicensed official repository is
  copied.
- Adaptation: four-channel genotype encoding, kernel sizes 4 and 20, residual
  dual stream, ISRU activation, bounded pooling, and one model per time point.

## WheatGP

- Upstream: <https://github.com/Breed-AI/WheatGP>
- Reviewed commit: `a061b8be1d231e21fc89ed7179ae58c2a3048819`
- License: Apache License 2.0.
- Adaptation: retains five SNP segments, a shared three-layer CNN with kernel
  sizes 1/3/9, LSTM hidden size 128, and dropout. Segment feature maps are
  adaptively pooled into a bounded token sequence before the LSTM. Upstream
  splitting, transfer learning, GUI, and pretrained checkpoints are excluded.

## Cropformer

- Upstream: <https://github.com/jiekesen/Cropformer>
- Reviewed commit: `e0a77cf699b034d956b249f41b2e1f357f486f06`
- License: MIT; upstream copyright is Hao Wang, Shen Yan, and Wei Fang.
- Adaptation: retains a one-dimensional CNN, multi-head attention, LayerNorm,
  dropout, and regression head. The supervised MIC selector, PED conversion,
  upstream random split, Optuna loop, and pretrained model are excluded. SNPs
  use train-standardized 0/1/2 dosage instead of the upstream PED-derived 0–9
  genotype categories and are tokenized with bounded adaptive pooling so
  attention remains tractable.
