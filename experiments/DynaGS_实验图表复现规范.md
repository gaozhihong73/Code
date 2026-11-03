# DynaGS 期刊论文实验图表复现规范

> 版本：v1.5（2026-11-02）  
> 用途：为后续运行实验、汇总结果并自动绘制论文图表提供唯一执行规范。图表中的所有数值必须来自正式实验输出；本文件中的数值、布局示意和结论方向均不是实验结果。

## 1. 论文图表总览

| 论文位置 | 编号 | 内容 | 类型 | 是否主文 |
|---|---|---|---|---|
| 方法 | Figure 1 | DynaGS总体架构、SNP筛选、Wide–Deep–Gate与动态输出 | 架构图 | 是 |
| E1+E2 | Figure 2 | 已知/未知QTL条件下的轨迹预测与筛选恢复 | 综合图 | 是 |
| E3 | Table 2 | 模拟数据上12种方法的整体预测精度 | 三线表 | 是 |
| E3 | Figure 3 | 配对性能差异及逐时间点预测能力 | 四面板图 | 是 |
| E4 | Table 3 | 真实数据上的交叉验证预测精度 | 三线表 | 是 |
| E4 | Figure 4 | 真实表型的泛化能力及个体外推预测 | 条件性综合图 | 是 |
| E5 | Table 4 | SNP筛选与下游预测器消融 | 三线表 | 是 |
| E5 | Figure 5 | Wide、Deep和Gate的结构消融 | 双面板森林图 | 候选主图 |
| 补充材料 | Tables S1–S4 | 完整统计推断、效率、敏感性与全量结果 | 表格 | 否 |
| 补充材料 | Figures S1–S4 | 全模型时序热图、个体轨迹、敏感性等 | 图 | 否 |

主文遵循“**表格给精确数值，图形给差异、动态规律和直观案例**”的原则。不要再额外绘制12个模型的三指标柱状图，也不要为了突出DynaGS而裁掉负差值、压缩坐标或选择最好看的测试样本。

## 2. 全局实验、统计和文件约定

### 2.1 统一实验单位

- **模拟实验（E1、E2、E3、E5）**：每一种实验设置独立重复 `R=10` 次。一个重复必须固定随机种子、训练/测试个体划分、QTL、表型噪声和所有模型使用的数据划分。
- **真实数据（E4）**：采用5次重复外层5折交叉验证。模型在同一外层测试折上比较；特征筛选、标准化、调参均只能在对应训练侧完成。
- **时间点**：模拟数据默认 `T=20`。真实数据使用真实采样时间与单位，不能强行改成等间隔的1–20。
- **配对原则**：DynaGS和任何比较方法在同一重复、同一外层测试折、同一测试个体上计算性能。配对差值是所有森林图和显著性检验的最小单元。

### 2.2 指标定义与方向

| 指标 | 含义 | 最优方向 | 计算建议 |
|---|---|---|---|
| PCC | 每个时间点在测试个体间计算Pearson相关，再对有效时间点取算术平均 | ↑ | 时间点PCC无定义时记为`N/A`并从该重复的时间平均中排除，同时记录有效时间点数；不得改用整体展平口径 |
| NRMSE | 归一化均方根误差 | ↓ | `RMSE / (训练集观测表型P95 − P5)`；分母仅由对应重复的训练个体拟合，不得使用测试集范围或标准差 |
| Time-adjusted R² | 考虑时间项后的拟合优度 | ↑ | 在代码中固定公式、时间编码、汇总方式；不要同普通R²混用 |
| QTL Recall@k | 前k个筛选SNP命中真实QTL的比例 | ↑ | 仅模拟数据可算；应报告精确命中和LD-aware命中 |
| Turn-point error | 预测与潜在真值主拐点时间之差 | ↓ | 仅模拟数据可算；预测与真值均以PCHIP插值至1,000个等距点，在去除两端各5%的内部区间取最大一阶导数时间；最大导数非正、落在内部边界或曲线振幅低于训练集尺度1%时记为无效 |

Mean基线若在每个时间点输出训练集均值，则跨测试个体的PCC无定义，所有PCC表和图中必须写作 `N/A`，不能填0。

### 2.3 统计推断

1. 对每个重复先计算模型指标，再计算配对差值；均值点估计在完整等权配对时可与两个均值之差相等，但SD、CI、效应量和检验必须基于重复内配对差值，不能基于两组独立汇总值。
2. 森林图点估计为10个重复配对差值的均值；95% CI由配对bootstrap得到。建议对10个配对差值做不少于10,000次重采样，并固定bootstrap随机种子。
3. 多重模型比较的p值采用Holm校正。主图不强制放p值；完整的原始p、Holm p、效应量和CI放入补充表。
4. E4中5折之间不视为独立重复。应先在每次重复内聚合其5个外层fold，再在5次重复层级进行配对汇总/推断，并在表注中明确。
5. 主表默认使用 `mean ± SD`。所有数值保留3位小数；若指标很小，保留4位但整列统一。最佳值加粗，次优值可加下划线。

### 2.4 推荐标准化结果文件

所有作图脚本应只读取规范化长表，避免从日志人工抄数。建议至少输出以下CSV（字段名可调整，但语义不可缺失）：

```text
results/metrics_by_repeat.csv
experiment,condition,repeat,fold,trait,model,timepoint,metric,value,n_test,seed

results/predictions_oof.csv
experiment,condition,repeat,fold,trait,model,sample_id,time,value_observed,value_predicted,value_latent,is_test

results/selection_by_repeat.csv
experiment,condition,repeat,method,rank,snp_id,is_causal,is_ld_hit,score

results/efficiency_by_repeat.csv
experiment,condition,repeat,fold,model,n_input_snps,n_parameters,screening_sec,training_sec,inference_sec,total_sec,peak_memory_mb,peak_vram_mb
```

模拟数据的`value_latent`可由同一`experiment/condition/repeat/sample_id/time`关联的`truth/true_curve.csv`提供；真实数据该字段必须为空。额外保存 `splits/*.csv`（训练/测试样本ID与group信息）及 `config/*.yaml`（种子、超参数、软件版本、硬件、数据版本）。每张图的元数据记录输入路径，必要时保存输入副本；不要求校验和。

### 2.5 通用视觉规范

- 输出：主图优先PDF/SVG矢量版，同时输出600 dpi PNG；单栏宽约85 mm、双栏宽约175–180 mm。
- 字体：Arial/Helvetica或期刊指定字体；最终缩放后文字不小于7 pt。数学符号使用规范斜体/正体。
- 颜色、marker和线型以 `experiments/experiment_plan.md` 第8.2节及 `dynags-figure/scripts/plot_style.py` 为唯一来源：DynaGS `#0072B2`，Cropformer `#AA4499`，XGBoost `#882255`，Random Forest `#CC79A7`，潜在真值 `#000000`，观测值 `#B3B3B3`，零线 `#9E9E9E`。同一对象跨所有图保持同色。
- 不用彩虹色、不用3D柱状图、不截断误差线、不用只靠颜色区分类别。线型或点形必须提供冗余区分。
- 图例只出现一次且模型排序固定：`DynaGS, Cropformer, XGBoost, Random Forest, 其他基线`。森林图中对比模型的排序固定并在E3与补充表保持一致。
- 图注必须写明：重复次数、误差线含义、样本划分、指标方向、正差值代表的含义，以及数据是模拟还是真实。

## 3. Figure 1：DynaGS方法架构图（方法部分）

### 3.1 目的

让读者在一张图内理解从高维SNP输入到动态表型预测的完整路径，而不是复述每层网络参数。

### 3.2 必须呈现的模块和流向

```text
Genotype matrix → training-only SNP screening → selected SNPs
                                     ↓
             normalized time + polynomial basis features
                                     ↓
       Wide branch ─┐
                    ├→ gated fusion → dynamic phenotype prediction at time t
       Deep branch ─┘
```

- 输入端：个体SNP基因型矩阵，以及时间点 `t`（或标准化时间与多项式基函数）。
- 筛选端：标注“仅在训练集内进行”，并表现GWAS/DC-SIS/组合策略及选取的Top-k SNP；不能把测试样本参与筛选。
- Wide分支：表现线性/低阶显式遗传效应与时间交互的作用，不需要绘制每一个神经元。
- Deep分支：表现SNP表示、非线性特征学习和时间条件信息；名称需与代码实际模块相同。
- Gate：展示由Wide/Deep特征学习的自适应融合权重，且明确它不是简单相加。
- 输出：`ŷ_i(t)`，对应个体i在任意观测时间的预测表型。

### 3.3 不应出现

- 未实现的注意力、线性注意力、外部环境变量或多任务模块。
- 训练集和测试集混合的筛选流程。
- 夸大“发现因果QTL”的措辞；筛选仅是候选SNP优先级。

## 4. E1+E2：模拟数据的动态预测与SNP筛选验证（Figure 2）

### 4.1 实验定位

E1验证已知因果位点（Oracle）条件下，模型能否学习动态遗传效应并恢复轨迹；E2验证真实可用的筛选（Screened）条件下，预测能力与QTL信号保留情况。两者应使用**相同模拟机制、相同评价指标和相同图形语言**，只改变输入SNP来源。

### 4.2 统一设置（必须写入图注或正文）

E1/E2完整设计为`P={5,000, 10,000} × NTR={100, 200, 400, 800, 1,600} × h²={0.1, 0.4, 0.8}`，每个条件10个独立重复。Figure 2主文只展示`P=10,000`的15个`NTR × h²`条件；`P=5,000`的相同全因子图、完整数值和敏感性结果进入补充材料。图注或正文须记录训练/测试样本数、QTL数、T=20、时间函数/多项式阶数、噪声机制、Oracle输入规则、Screened最终20 SNP及训练侧筛选规则。

### 4.3 Figure 2最终布局

Figure 2必须同时给出微观曲线、全因子宏观规律和筛选恢复质量；不得将主文宏观结果缩减为单一参考条件的10次重复点图。

| 面板 | 内容 | 数据/绘制方法 | 核心问题 |
|---|---|---|---|
| A | Oracle与Screened的代表个体轨迹 | 固定参考条件`P=10,000, NTR=400, h²=0.8`；3个预先选定测试个体各一个小图，叠加潜在真值、观测值、Oracle预测和Screened预测，并标出两种预测的最大生长速率时间及其相对真值的偏差 | 筛选是否改变个体动态曲线与主拐点恢复 |
| B | 全因子预测性能 | `P=10,000`下按`h²=0.1/0.4/0.8`分三列，PCC和NRMSE分两行；横轴NTR=100,200,400,800,1600，Oracle与Screened以固定线型/点形区分。每点为10重复均值，重复级95% CI为误差带或误差线，并显示低透明度原始重复点 | 不同训练样本量和遗传力下筛选造成的预测变化 |
| C | Exact Recall@20 热图 | `P=10,000`、5个NTR×3个h²条件；横轴NTR，纵轴h²，每格为10重复均值，格内标注均值，色标范围固定为[0,1] | 真实QTL的精确恢复随样本量和遗传力的变化 |

LD-aware Recall@20仅在与Exact Recall@20提供不同信息时作为D面板保留。若两者在Figure 2范围内逐重复完全相同，则删除D面板，不为维持面板数量而重复绘图；LD-aware定义、核查结果和完整数值转入补充表或图注。

Figure 2采用紧凑的三行整体版式：A、B、C三个主区域必须共享相同的左、右绘图区边界，A与B的三列必须等宽且列中心一致；A、B、C面板字母使用统一的图坐标横位置并在各区域顶部对齐。C面板色标置于共享绘图区右侧的独立窄轴中，不得压缩C而造成三个主区域宽度不一致。纵向间距仅保留标题、刻度和轴标签所需空间，不得出现大块无信息留白。

P=5,000结果不得混入Figure 2主文面板，也不得因表现更好或更差而选择性展示；其对应全因子性能图和Recall@20热图进入补充材料。

### 4.4 代表轨迹和拐点的精确规则（A）

- 参考条件固定为`P=10,000, NTR=400, h²=0.8`。在10个重复中，先选取Screened观测PCC最接近该条件10次重复中位数的重复；再在该重复测试集内按观测时间均值分为低、中、高三层，每层选取最接近该层中位数的一个样本。记录repeat、sample_id、sample_idx和层级，禁止按预测效果或曲线美观程度挑选。
- 横轴为真实时间；纵轴为模拟表型值。三幅小图固定时间范围、纵轴范围、样本顺序和唯一图例。
- 潜在真值为黑色细实线，观测值为浅灰散点，Oracle为DynaGS蓝色实线空心圆点，Screened为同色虚线方点；不得只靠颜色区分Oracle与Screened，也不得通过缩窄纵轴人为放大两种预测的差异。
- 对潜在真值、Oracle和Screened曲线分别使用第2.2节的PCHIP拐点协议，显示最大生长速率时间的竖线或点标记。以数学格式`Δt_O = t_O − t_truth`、`Δt_S = t_S − t_truth`标注两种预测相对真值的偏差；无有效拐点时明确标记`N/A`，不得补零。时间范围1–20使用1,000点网格时相邻点约为0.019，因此图中的0.02级偏差可能只对应一个插值网格间隔，不得解释为真实生物学时间测量精度。
- 三个代表个体仅用于直观展示，不能代替总体拐点统计。全体测试个体的拐点MAE、有效拐点比例、有效个体数、曲线CCC和个体曲线NRMSE进入补充表。

### 4.5 全因子性能与筛选恢复规则（B–C）

- B仅使用观测表型主指标：逐时间点跨测试个体PCC的平均值，以及以对应重复训练集P95−P5归一化的NRMSE。每个条件的10个重复为唯一独立统计单位；线和大标记为10重复均值，误差棒为重复级bootstrap 95% CI，浅色小点为10个独立模拟重复。Oracle和Screened使用相同蓝色但不同线型与点形。PCC若未展示完整`[-1,1]`，图注和元数据必须记录实际显示范围且不得裁掉任何原始点或CI。
- C只报告正式固定20 SNP输入的Exact Recall@20，不绘制不存在的多个k曲线。每格必须由标准化正式实验结果表中的10次重复汇总，不得读取演示结果或中间筛选缓存；完整原始点、CI和LD-aware Recall@20进入补充材料。
- LD-aware hit仍须以同一重复训练个体的剂量相关估计，匹配阈值固定为`r² ≥ 0.8`。绘图前逐重复核查`exact_recall`与`ld_aware_recall`；完全相同时主图只保留C，并在元数据和图注中记录该事实。Oracle不绘制筛选Recall。
- C的NTR横轴顺序为100、200、400、800、1600，h²纵轴顺序为0.1、0.4、0.8；色标固定为[0,1]并使用`viridis`。

### 4.6 Figure 2图注模板

`Figure 2. Dynamic phenotype prediction and marker recovery across simulated settings with P = 10,000. (A) Out-of-sample trajectories for pre-specified low-, middle-, and high-level test individuals under a median-representative repeat of P = 10,000, NTR = 400, and h² = 0.8. Black lines denote latent noise-free trajectories, grey points denote noisy observations, and solid open-circle versus dashed-square blue lines denote Oracle versus training-only Screened DynaGS predictions. Δt_O and Δt_S are the predicted maximum-growth-rate times minus the latent-truth time under the pre-specified 1,000-point PCHIP protocol; 0.02 is approximately one interpolation-grid interval and does not imply biological timing precision. Overall inflection MAE and valid-individual counts are reported in the Supplementary Material. (B) PCC, averaged over valid timepoint-wise correlations across test individuals, and NRMSE, normalized by the corresponding training-set P95−P5, across five NTR levels and three heritabilities. Lines and large markers are means of 10 independent repeats, error bars are repetition-level bootstrap 95% CIs, and pale points are individual simulation repeats; displayed PCC limits contain every raw point and CI. (C) Exact Recall@20 from the formal E2 result table. LD-aware Recall@20, defined using training-individual r² ≥ 0.8, was identical to Exact Recall@20 in all P = 10,000 repetitions and is therefore reported in the Supplementary Material rather than duplicated here. P = 5,000 results are also reported in the Supplementary Material.`

## 5. E3：模拟数据完整模型比较

### 5.1 固定实验条件

E3是完整系统比较，不可把DynaGS的优势全部归因于Wide–Deep结构：DynaGS使用筛选后SNP，而多数基线输入10,000 SNP；结构归因必须由E5完成。建议在Table 2表注中固定说明：SNP=10,000、NTR=400、h²=0.8、QTL=20、T=20、独立重复=10（以实际最终配置为准）。

比较模型及固定顺序：Mean、Ridge-All、Ridge-A+D、GBLUP-All、RR-GBLUP、Random Forest、XGBoost、DNNGP-SNP、DLGWAS、WheatGP、Cropformer、DynaGS。

### 5.2 Table 2：12种方法整体预测性能

| Model | PCC ↑ | NRMSE ↓ | Time-adjusted R² ↑ |
|---|---:|---:|---:|
| Mean | N/A | `mean ± SD` | `mean ± SD`或N/A（按定义） |
| Ridge-All | `mean ± SD` | `mean ± SD` | `mean ± SD` |
| Ridge-A+D | … | … | … |
| GBLUP-All | … | … | … |
| RR-GBLUP | … | … | … |
| Random Forest | … | … | … |
| XGBoost | … | … | … |
| DNNGP-SNP | … | … | … |
| DLGWAS | … | … | … |
| WheatGP | … | … | … |
| Cropformer | … | … | … |
| **DynaGS** | … | … | … |

表注必须说明：均值和SD来自10次独立重复；最优值加粗；PCC对Mean无定义；完整CI、配对检验与Holm校正在Table S1。若R²允许为负，保留负值，不能截为0。

Table 2按冻结E3条件从12模型×10重复的`metrics.json`直接读取`overall.pcc`、`overall.nrmse`和`overall.time_adjusted_r2`，以重复为单位计算均值和样本SD；不通过`run_manifest.json`或预测文件重算该表指标。输出可核验的CSV、Markdown预览和LaTeX三线表。

### 5.3 Figure 3：四面板综合图

#### A. DynaGS相对基线的ΔPCC森林图

定义：`ΔPCC = PCC_DynaGS − PCC_baseline`。只包括PCC可定义的10个有效基线，不包括Mean。纵轴列出基线模型，横轴为差值；0位置使用浅灰虚线。点为10次配对重复的均值，水平线为配对bootstrap 95% CI。正值代表DynaGS更好，负值代表基线更好。模型按ΔPCC均值或预先固定的模型类型排序，但不要按显著性筛选。

A、B均在首行显式标出`DynaGS (reference)`，以零点菱形表示比较参照；该行不是额外的基线，也不绘制没有统计意义的自我比较CI。

#### B. DynaGS相对基线的ΔNRMSE森林图

定义：`ΔNRMSE = NRMSE_baseline − NRMSE_DynaGS`。包含Mean在内的11个基线。其余绘制方式与A相同；正值仍代表DynaGS更好。A、B横轴都必须包含0，并尽可能使用视觉上对称的范围。

#### C. PCC over Time

仅展示4个预先指定代表模型：DynaGS、XGBoost、Random Forest、Cropformer。横轴Time 1–20，纵轴时间点PCC。每条线是10重复均值；误差带为95% CI（推荐）或SD（如使用必须明确）。曲线平稳时每隔若干点显示marker，避免20个点全部堆叠。所有时间点的模型数据和12模型热图进入补充材料。

#### D. NRMSE over Time

与C完全相同的4个模型、时间点、颜色、测试划分及误差表示；纵轴改为NRMSE。不要因某模型表现不佳临时替换代表模型。

### 5.4 Figure 3版式和输出

- 推荐2×2双栏版式：A、B在上排；C、D在下排。
- A、B共享“positive favours DynaGS”文字提示；C、D共享图例。
- 不绘制12模型×三指标柱状图；它与Table 2重复且不利于读取近似值。
- 图脚本输出：`fig3_model_comparison.pdf/png`，并导出对应的`fig3_pairwise_effects.csv`和`fig3_timewise_metrics.csv`。
- Figure 3当前复现以E3正式运行目录内各模型、各重复的`metrics.json`作为指标值唯一输入：A–B取`overall.pcc`及`overall.nrmse`后在同一重复内计算差值，C–D取`by_time.pcc`及`by_time.nrmse`；不得通过`run_manifest.json`或预测文件重新计算绘图指标。已有异常源文件须在明确核实后单独修复并记录，不能在绘图时静默改值。审稿阶段仅生成PNG，确认版式后再输出其他格式。

### 5.5 E3补充材料

- **Table S1**：每个DynaGS–基线比较的差值均值、SD、95% CI、标准化效应量、原始p和Holm p。
- **Table S2**：12种模型输入SNP数、可训练参数量（不适用写N/A）、筛选耗时、训练耗时、总端到端耗时、峰值内存/显存。DynaGS总耗时含训练侧筛选；单时间点模型必须统计20个独立预测器的总训练成本。
- **Figure S1**：12模型×20时间点的PCC/NRMSE热图（可选）。色条范围、模型顺序和时间顺序固定，PCC和NRMSE分开色条。
- **Figure S2**：固定参考重复中1–2个测试个体的多模型轨迹（真实潜在曲线、DynaGS、XGBoost、RF、Cropformer），用于解释总体指标相近但阶段偏差不同的情形。

## 6. E4：真实数据模型对比与泛化验证

### 6.1 实验定位与严格边界

E4证明真实动态表型上的适用性，而不是重复E3的模型排名。真实数据没有已知QTL真值和潜在无噪声轨迹，因此不得报告QTL Recall、潜在轨迹恢复准确率或模拟转折点误差。任何生物学拐点分析只能是基于观测/预测曲线的描述性估计。

真实数据源、性状数、时间点、样本规模、家系/群体/环境分组尚需冻结。若存在群体、家系或环境泄漏风险，优先采用grouped CV，并保存每一折的组分配。

### 6.2 Table 3：真实数据整体性能

#### 单一主要动态表型

表结构与Table 2一致，列为Model、PCC、NRMSE、Time-adjusted R²，报告5次重复的外层CV汇总结果。每次重复内先对5个fold按明确权重聚合，再计算跨重复mean ± SD。

#### 多个动态表型

- 主表展示2–4个预先选定主性状的PCC，或采用每个性状一组PCC/NRMSE/R²子列；只能在版面允许时使用。
- 必须事先声明主性状选择依据（育种重要性、观测完整度或预注册优先级），不能按DynaGS结果选。
- 若报告跨性状总体均值，说明等权还是按测试观测数加权；不同时间点/缺失率不可无说明地混合。
- 其余“性状×模型×指标”完整结果放Table S3。

### 6.3 Figure 4：按真实表型数量选择布局

#### 多表型方案（推荐）

| 面板 | 内容 | 规则 |
|---|---|---|
| A | 不同性状的配对ΔPCC | DynaGS分别对XGBoost和Cropformer；每个性状给点估计和95% CI，坐标包含0 |
| B–D | 三个代表个体的外层测试预测轨迹 | 低、中、高表型层级；观测点+DynaGS+一个预先指定代表基线 |

对A，定义 `ΔPCC_p^(m) = PCC_DynaGS,p − PCC_m,p`。同一表型、同一外层划分配对。可将vs XGBoost和vs Cropformer做成上下小分面或同一图中分组点，不应遮挡。

#### 单一表型方案

不画“跨性状改善”面板。主文Figure 4仅保留三个代表个体轨迹；若正式结果显示显著的早中晚期性能差异，可用逐时间点性能图替换信息量低的个体面板，但需在实验冻结前决定。

### 6.4 真实个体轨迹精确规则（Figure 4B–D）

- 只使用**out-of-fold**预测：该样本对应模型训练过程从未见过该样本。图注明确写出。
- 样本选择：按每名测试个体的观测值时间平均值（或预先定义的面积/末期值）分低、中、高三层；在固定参考重复中抽取。保存`sample_id`、repeat、fold和层级。
- 灰色散点为真实观测值；DynaGS蓝色实线；代表深度基线（默认Cropformer）紫色虚线。若预先冻结XGBoost为基线，则全图统一替换，不能事后按谁更弱决定。
- 横轴为实际采样时间及单位；纵轴为真实性状名称和单位。因测量时间不规则，预测曲线可在密集网格计算后显示，但观测点必须保留原时间。
- 同一个个体的两模型预测必须来自同一次重复/同一外层fold。不要将不同重复的预测平均后伪装为单个独立测试预测；如使用重复平均，图注应明确并另存逐重复曲线。
- 不画不存在的潜在真值线；不把预测曲线平滑当成生物学真实曲线。

### 6.5 E4补充材料

- **Table S3**：所有性状×模型×指标、外层重复汇总和统计检验。
- **Table S4**：真实数据模型效率与资源消耗；字段同E3 Table S2。
- **Figure S3**（可选）：逐真实时间点PCC/NRMSE；真实时间不等间隔时横坐标按连续日期/积温显示。
- 训练曲线、所有个体轨迹、大量性状分面均只可作补充，不进入主文。

## 7. E5：筛选贡献、结构消融与敏感性

### 7.1 实验定位

E5拆分“筛选带来的收益”和“网络结构带来的收益”。这是对DynaGS创新点的因果归因实验，须控制数据划分、输入SNP数和训练预算。不要用E3的完整系统比较替代E5。

### 7.2 Table 4：SNP筛选与下游预测器（必须）

| Pipeline | Input SNPs | PCC ↑ | NRMSE ↓ | Time-adjusted R² ↑ | ΔPCC vs full |
|---|---:|---:|---:|---:|---:|
| Full DynaGS | Screened Top-k | `mean ± SD` | `mean ± SD` | `mean ± SD` | Reference |
| DynaGS without screening | All SNPs | … | … | … | … |
| Screened + Ridge | Same Top-k | … | … | … | … |
| Screened + Random Forest | Same Top-k | … | … | … | … |
| Screened + XGBoost | Same Top-k | … | … | … | … |

关键控制条件：

- `Screened +`各模型必须使用**该重复训练集内由DynaGS筛选流程产生的同一批Top-k SNP**；测试侧不可参与。
- “DynaGS without screening”必须明确是否因显存/参数限制调整了网络；若调整，就不是纯筛选消融，应另列说明。
- ΔPCC定义为`PCC_full − PCC_pipeline`，正值表示完整DynaGS更好。表注中写出配对CI位置（主表或Table S1）。
- 表格优先于柱状图，因仅5条管线且其功能是精确的归因比较。

### 7.3 Figure 5：网络组件结构消融（候选主图）

#### 比较版本

| 版本 | 结构 |
|---|---|
| all | Wide + Deep + Gate（完整DynaGS） |
| wide | 仅Wide通路 |
| deep | 仅Deep通路 |
| add | Wide + Deep直接相加，无Gate |

实验分别在S1（Oracle）与S2（Screened）条件下执行，每组10独立重复；其他超参数、训练/测试划分、训练轮数和优化策略保持一致。

#### 面板A：ΔPCC；面板B：ΔNRMSE

- 对每个条件和版本，`ΔPCC = PCC_all − PCC_ablation`。
- `ΔNRMSE = NRMSE_ablation − NRMSE_all`。
- 两种定义的正值均表示完整模型更好。
- 纵轴：`wide`、`deep`、`add`；颜色/点形：S1 Oracle、S2 Screened。横轴均应有0灰虚线。
- 每个条件×版本一个均值点和配对bootstrap 95% CI；可轻微上下错位避免重叠，不可把S1/S2混成一个均值。
- `all vs add`是Gate有效性的直接检验。若CI跨0或增益很小，结果和论文表述必须承认Gate贡献有限/不稳定。

**是否保留主文Figure 5的决策规则**：当S1与S2下各组件贡献模式不同、或CI提供了表格无法直观看出的信息时保留；若所有结果简单且完全一致，可删除Figure 5，将对应数值扩展进入Table 4或补充表。

### 7.4 E5c：敏感性分析（补充材料）

1. **多项式阶数**：`degree = 0, 1, 3, 5, 7`。补充表给PCC、NRMSE、训练失败率/方差；只有出现明确单调或阈值规律时才画折线图。
2. **筛选策略**：GWAS-only、DC-SIS-only、串联、并集、随机选择。报告输入位点数、Recall@20、LD-aware Recall@20、PCC、NRMSE。不同策略必须控制最终位点数，或明确位点数是实验变量。
3. 随机选择至少重复相同次数，并固定每个重复的随机种子；不得以单次随机选择作为基线。

## 8. 图表生成前的验收清单

### 8.1 数据完整性

- [ ] 每个图表的`repeat`数与方案一致；没有缺失重复被静默删除。
- [ ] 同一对比中的模型共享完全相同的测试样本、时间点、重复和fold。
- [ ] 所有筛选、缩放和调参仅使用训练侧数据。
- [ ] E4个体图全部为out-of-fold预测，且样本ID和fold可追溯。
- [ ] Mean的PCC为N/A；真实数据不含模拟真值/QTL指标。
- [ ] NRMSE分母、PCC汇总口径、Time-adjusted R²公式全程一致。

### 8.2 统计与可视化

- [ ] 表内均值、SD、最佳值标记由脚本自动生成，非手工修改。
- [ ] 森林图差值方向在标题、横轴和图注中一致，正值均明确代表DynaGS更好。
- [ ] CI为配对bootstrap 95% CI，且种子和重采样次数记录。
- [ ] 坐标轴含0线、不会截掉负效应或CI。
- [ ] 代表个体按预定义分层规则选择，非按最优预测筛选。
- [ ] 颜色、模型名称、缩写和模型顺序在Figure 1–5、表格及补充材料中完全一致。

### 8.3 交付目录建议

```text
paper_outputs/
  tables/
    table2_simulation_comparison.csv
    table3_realdata_comparison.csv
    table4_ablation_screening.csv
    tableS1_pairwise_statistics.csv
    tableS2_simulation_efficiency.csv
    tableS3_realdata_all_traits.csv
    tableS4_realdata_efficiency.csv
  figures/
    fig1_architecture.pdf
    fig2_oracle_screened.pdf
    fig3_model_comparison.pdf
    fig4_realdata_validation.pdf
    fig5_architecture_ablation.pdf
    figS1_all_model_heatmap.pdf
    figS2_simulated_individual_trajectories.pdf
    figS3_realdata_timewise_metrics.pdf
  metadata/
    selected_individuals.csv
    experimental_settings.yaml
    figure_provenance.json
```

## 9. 结果叙事与禁止性表述

- E1/E2：说明Oracle与Screened下的动态预测和信号保留，不把筛选命中称为因果验证。
- E3：先报告整体性能（Table 2），再谈配对差异与稳定性（Fig.3A–B），然后谈时间阶段表现（Fig.3C–D），最后简述效率（补充表）。若XGBoost或其他模型在某项指标更好，必须如实写出。
- E4：强调真实数据泛化、不同性状差异和out-of-fold个体预测；不要将观测值与平滑预测的视觉贴合写成“恢复真实连续生物过程”。
- E5：用Table 4说明筛选是否可迁移，用Fig.5说明Wide、Deep、Gate的额外贡献。不能在未对比`add`的情况下声称门控融合有效。
- 不使用“显著优于所有模型”“全面领先”等绝对措辞，除非预定义所有主指标、所有比较和Holm校正结果均支持；更稳妥的是具体写明哪一指标、哪一条件、效应量及CI。

## 10. 实施顺序（供Codex执行）

1. 冻结实验配置与模型版本，生成/检查上述4类标准化结果CSV。
2. 先运行完整性检查和指标重算脚本
3. 生成E1/E2图，固定代表样本ID；将该ID清单复用于相关补充图。
4. 生成Table 2和Figure 3，先核对所有差值方向与Table 2数值一致。
5. 在真实数据结构冻结后，按单/多表型分支生成Table 3和Figure 4。
6. 生成Table 4；根据结构消融是否具备足够信息决定是否保留Figure 5。
7. 导出补充图表，进行字体、颜色、面板编号、图注和600 dpi检查。
8. 将每张图的输入数据、脚本版本、随机种子和生成时间写入`figure_provenance.json`，使其可一键重建。
