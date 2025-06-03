from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from PIL import Image
import glob

# 禁用LaTeX渲染，使用matplotlib的mathtext
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'


def main():
    # === 1. 读取数据 ===
    txt_path = Path(r"新建 文本文档.txt")          # 文件名含空格，故用原始字符串或 Path
    df = pd.read_csv(txt_path, sep="\t")

    # 使用实际的数据列名，但显示用数学格式
    actual_x_col, actual_y_col = "bi1", "bi2"  # 实际数据列名
    display_x_col, display_y_col = r"$l_{01}$", r"$l_{12}$"  # 显示用的数学格式
    z_cols = df.columns[2:]                       # 第 3~5 列

    # === 2. 生成规则网格坐标 ===
    nx, ny = 200, 200                             # 网格分辨率
    xi = np.linspace(df[actual_x_col].min(), df[actual_x_col].max(), nx)
    yi = np.linspace(df[actual_y_col].min(), df[actual_y_col].max(), ny)
    XX, YY = np.meshgrid(xi, yi)                  # 规则网格

    # 散点坐标
    points = df[[actual_x_col, actual_y_col]].values

    # === 3. 对每个 z 列做插值并绘图 ===
    for z_col in z_cols:
        # ① 把散点转换为规则网格矩阵
        pivot = df.pivot_table(index=actual_y_col,
                               columns=actual_x_col,
                               values=z_col)

        # ② 取网格坐标
        X = pivot.columns.values        # bi1
        Y = pivot.index.values          # bi2
        XX, YY = np.meshgrid(X, Y)
        ZZ = pivot.values               # (len(Y), len(X))

        # ③ 只用线性插值补齐（不做最近邻填补）
        if np.isnan(ZZ).any():
            ZZ = pd.DataFrame(ZZ, index=Y, columns=X).interpolate(
                method="linear", axis=1).interpolate(
                method="linear", axis=0).values
            # 不再用最近邻填补，NaN 保持为 NaN

        # ④ 裁剪 >0.1
        ZZ_clipped = np.minimum(ZZ, 0.1)

        # === 5. 绘制等值面 ===
        plt.figure(figsize=(8, 6))
        
        # 创建等值线级别（0到0.1）
        levels = np.linspace(0, 0.1, 50)
        
        # 使用冷色调色带
        contour = plt.contourf(XX, YY, ZZ_clipped, levels=levels, cmap="viridis", 
                              vmin=0, vmax=0.1, extend="neither")
        
        # 绘制黑色等值线并显示数值
        contour_line = plt.contour(XX, YY, ZZ, levels=[0.1], colors=['black'], 
                                  linewidths=2, linestyles='-')
        plt.clabel(contour_line, inline=True, fontsize=10, fmt='%.1f', colors='black')
        
        # 创建颜色条，设置每隔0.02显示一个刻度
        cbar = plt.colorbar(contour, label=f"|{z_col}|")
        cbar.ax.tick_params(labelsize=11)
        
        # 设置颜色条刻度：每隔0.02显示
        tick_positions = np.arange(0, 0.1 + 0.02, 0.02)  # 0, 0.02, 0.04, 0.06, 0.08, 0.10
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in tick_positions])
        
        # 设置标签（使用mathtext格式，不显示标题）
        plt.xlabel(display_x_col, fontsize=14)
        plt.ylabel(display_y_col, fontsize=14)
        
        # 设置坐标轴范围
        plt.xlim(df[actual_x_col].min(), df[actual_x_col].max())
        plt.ylim(df[actual_y_col].min(), df[actual_y_col].max())
        
        plt.tight_layout()

        # === 6. 保存图像 ===
        img_name = f"{z_col}_contour_clipped_01.png".replace("/", "_")
        plt.savefig(img_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {img_name}")


def rgb2gray(img_arr):
    """
    如果是彩色 (H, W, 3) 或 (H, W, 4) → 转灰度 (H, W)
    """
    if img_arr.ndim == 2:  # 已经是灰度
        return img_arr.astype(float) / 255.
    # 若有 alpha 通道只取前 3 个
    img_arr = img_arr[..., :3]
    # ITU-R BT.601 加权系数
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(float) / 255.

def extract_grayscale_stats():
    # 1. 找到所有目标图片
    png_paths = sorted(glob.glob("*_contour_clipped_01.png"))
    if not png_paths:
        print("⚠️ 未找到 *_contour_clipped_01.png 相关文件")
        return

    rows = []
    for p in png_paths:
        # 2. 读图并转灰度
        img = Image.open(p).convert("RGBA")  # 保险起见先转 RGBA
        gray = rgb2gray(np.array(img))

        # 3. 计算统计量
        mean_val = gray.mean()
        std_val = gray.std()
        # 4. 直方图（可选）
        hist, bin_edges = np.histogram(gray, bins=50, range=(0, 1), density=True)

        rows.append({
            "filename": Path(p).name,
            "mean_gray": mean_val,
            "std_gray": std_val,
            "hist_bins": bin_edges[:-1],   # 可选
            "hist_vals": hist              # 可选
        })

        print(f"{p}: mean={mean_val:.4f}, std={std_val:.4f}")

    # 5. 保存 CSV（不含直方图列亦可）
    df = pd.DataFrame(rows).drop(columns=["hist_bins", "hist_vals"])
    df.to_csv("grayscale_stats.csv", index=False, float_format="%.6f")
    print("✅ 统计结果已保存到 grayscale_stats.csv")


if __name__ == "__main__":
    main()
    extract_grayscale_stats()