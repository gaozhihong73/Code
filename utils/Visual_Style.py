from typing import Literal

import matplotlib

# ==============================================================================
# 1. 强制设置后端 (Linux 无界面环境必备)
# ==============================================================================
# 注意：必须在 import pyplot 之前设置，否则在服务器上会报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 2. 定义统一的学术绘图风格
# ==============================================================================
ContextType = Literal["paper", "notebook", "talk", "poster"]


def set_style(
    context: ContextType = "paper", font_scale: float = 1.2, palette: str = "deep"
):
    """统一初始化绘图风格。

    Parameters
    ----------
    context : str, optional
        seaborn 上下文, 可选 ``"paper"`` (论文)、``"notebook"``、
        ``"talk"``、``"poster"``, 默认 ``"paper"``。
    font_scale : float, optional
        字体缩放比例, 默认 ``1.2`` 适合大多数论文插图。
    palette : str, optional
        配色方案, 默认 ``"deep"``。
    """

    # --- A. 基础字体设置 (核心需求: Times New Roman) ---
    # 定义字体优先级列表：首选 Times New Roman，如果没有安装，则回退到开源替代品
    font_list = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]

    plt.rcParams.update(
        {
            # 字体设定
            "font.family": "serif",
            "font.serif": font_list,
            "axes.unicode_minus": False,  # 解决负号显示为方块的问题
            # 分辨率与线条
            "figure.dpi": 300,  # 默认高DPI，保证导出清晰
            "axes.linewidth": 1.2,  # 坐标轴线宽，更清晰
            "grid.linestyle": "--",  # 网格线样式
            "grid.alpha": 0.3,  # 网格线透明度
            # 刻度朝向 (期刊通常要求刻度朝内)
            "xtick.direction": "in",
            "ytick.direction": "in",
            # 图例设置
            "legend.frameon": False,  # 图例默认去掉边框，更简洁
            "legend.fontsize": 10,
        }
    )

    # --- B. Seaborn 主题设置 ---
    # style="ticks": 带有刻度线的白色背景，是最标准的学术图表风格
    sns.set_theme(style="ticks", rc={"font.family": "serif", "font.serif": font_list})

    # 设置上下文 (自动调整线条粗细和字体大小)
    # seaborn.set_context 接受特定的字符串常量或一个 dict，这里确保传入值为合法的预设之一
    valid_contexts = {"paper", "notebook", "talk", "poster"}
    if isinstance(context, str) and context in valid_contexts:
        ctx = context
    else:
        # 如果传入不合法，则回退到默认 'paper'
        ctx = "paper"
    sns.set_context(ctx, font_scale=font_scale)

    # 设置默认色盘 (适合科研的配色)
    sns.set_palette(palette)

    print(
        f">>> [Visual Style] 绘图环境已配置: Backend=Agg, Font=Times New Roman, Context={context}"
    )


# 导出这些对象，供其他脚本直接使用
__all__ = ["plt", "set_style", "sns"]
