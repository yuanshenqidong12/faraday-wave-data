"""
容器区域检测脚本 — 用于 Faraday 波实验图像分析
======================================================
功能：从俯拍液面图像中自动识别圆形容器边界，输出圆心坐标和半径。
适用：圆柱玻璃容器 + 液态金属 + NaOH 溶液，俯拍视角。
原理：
  1. 用低亮度阈值 (>3/255) 找出所有"非纯黑背景"像素
  2. 计算这些像素的空间质心 → 容器圆心
  3. 沿径向扫描亮度梯度，找到容器内→背景的亮度断崖 → 容器半径
  4. 生成标注图供人工目视确认

使用方法：
  python container_detector.py <图片路径或文件夹路径>

  如果是文件夹，会自动选曝光最好的一张图来分析。

输出：
  - container_detected.png : 标注了检测结果的图片（用于人工验证）
  - 控制台输出：建议的 cx, cy, r 参数
"""

import sys, os, math
from PIL import Image, ImageDraw
import numpy as np


def find_container_center(img_array, brightness_threshold=3):
    """
    通过亮像素质心定位容器圆心。
    背景几乎是纯黑 (0-2)，液面区域略亮 (>3)。
    """
    h, w = img_array.shape
    ys, xs = np.where(img_array > brightness_threshold)

    if len(xs) < 100:
        raise ValueError(f"亮像素太少 ({len(xs)})，图像可能全黑，请检查曝光")

    cx = np.mean(xs)
    cy = np.mean(ys)
    return cx, cy


def find_container_radius(img_array, cx, cy):
    """
    沿径向从圆心向外扫描，找到亮度急剧下降的位置 = 容器边界。
    返回建议半径。
    """
    h, w = img_array.shape
    max_r = min(cx, cy, w-cx, h-cy) - 5

    # 径向平均亮度曲线
    radial_mean = []
    for r in range(20, int(max_r), 2):
        ring_vals = []
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))
            if 0 <= x < w and 0 <= y < h:
                ring_vals.append(img_array[y, x])
        if ring_vals:
            radial_mean.append((r, np.mean(ring_vals)))

    if not radial_mean:
        raise ValueError("无法计算径向亮度曲线")

    radii = np.array([x[0] for x in radial_mean])
    means = np.array([x[1] for x in radial_mean])

    # 找亮度断崖：mean 从 >10 降到 <5 的位置
    # 计算梯度
    gradient = -np.gradient(means)  # 亮度下降 = 正梯度

    # 在合理范围内找最大梯度（r > 100px 避免中心热点干扰）
    search_start = np.searchsorted(radii, 100)
    if search_start < len(gradient):
        peak_idx = search_start + np.argmax(gradient[search_start:])
        edge_r = radii[peak_idx]
    else:
        edge_r = radii[-1]

    # 容器内亮度通常在 10-200，外圈 < 5
    # 找到最后一个 mean > 背景阈值 的位置
    bg_threshold = 5
    for i in range(len(radii)-1, -1, -1):
        if means[i] > bg_threshold:
            inner_edge = radii[i]
            break
    else:
        inner_edge = edge_r

    # 取梯度法和阈值法的较大者，再加一点余量（容器外壁）
    suggested_r = int(max(edge_r, inner_edge)) + 3

    return suggested_r, radii, means


def validate_on_best_exposed(image_dir):
    """
    在文件夹中找曝光最好的图片进行分析。
    判断标准：容器区域内平均亮度最接近 100（目标曝光）。
    """
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if not files:
        raise FileNotFoundError(f"文件夹 {image_dir} 中没有图片文件")

    # 先用中间帧快速估计
    mid_file = files[len(files)//2]
    mid_path = os.path.join(image_dir, mid_file)
    test_img = Image.open(mid_path).convert('L')
    test_arr = np.array(test_img, dtype=np.float64)

    # 初检圆心
    cx0, cy0 = find_container_center(test_arr)

    # 从所有文件中选曝光最好的（容器内 mean 最接近 100）
    best_file = mid_file
    best_score = 999
    for f in files[::max(1, len(files)//20)]:  # 采样检查，加快速度
        fp = os.path.join(image_dir, f)
        img = Image.open(fp).convert('L')
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape

        # 粗略容器区域
        mask = np.zeros_like(arr, dtype=bool)
        for y in range(h):
            for x in range(w):
                if (x-cx0)**2 + (y-cy0)**2 < (min(cx0,cy0,w-cx0,h-cy0)*0.7)**2:
                    mask[y, x] = True

        if mask.sum() > 1000:
            roi_mean = arr[mask].mean()
            score = abs(roi_mean - 100)
            if score < best_score:
                best_score = score
                best_file = f

    return os.path.join(image_dir, best_file)


def annotate_and_save(img_path, cx, cy, r, output_path):
    """在图像上标注检测结果并保存"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 容器圆
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 255, 0), width=3)
    # 圆心十字
    draw.line([(cx-30, cy), (cx+30, cy)], fill=(255, 255, 0), width=2)
    draw.line([(cx, cy-30), (cx, cy+30)], fill=(255, 255, 0), width=2)
    # 中心热点排除区
    r_hot = 20
    draw.ellipse([cx-r_hot, cy-r_hot, cx+r_hot, cy+r_hot], outline=(255, 0, 0), width=2)
    # 参考圆（±5px 微调）
    for dr in [-5, 5]:
        draw.ellipse([cx-(r+dr), cy-(r+dr), cx+(r+dr), cy+(r+dr)], outline=(100, 100, 100), width=1)

    # 标注信息
    draw.rectangle([4, 4, 320, 60], fill=(0, 0, 0))
    draw.text((8, 8),  f'DETECTED: cx={cx:.1f}  cy={cy:.1f}  r={r}', fill=(0, 255, 0))
    draw.text((8, 24), f'GREEN = container boundary (r={r})', fill=(0, 255, 0))
    draw.text((8, 40), f'GRAY  = +/-5px fine-tune reference (r={r-5}, {r+5})', fill=(150, 150, 150))
    draw.text((8, 56), f'RED   = center hotspot exclusion (r=20)', fill=(255, 0, 0))

    img.save(output_path)
    print(f"标注图已保存: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]

    # 判断是文件还是文件夹
    if os.path.isfile(target):
        img_path = target
    elif os.path.isdir(target):
        print(f"正在从文件夹中选择曝光最好的图片...")
        img_path = validate_on_best_exposed(target)
        print(f"  选中: {os.path.basename(img_path)}")
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)

    print(f"分析图片: {img_path}")

    # 读取图像
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape
    print(f"图像尺寸: {w}x{h}")

    # === 步骤 1: 检测圆心 ===
    cx, cy = find_container_center(arr)
    print(f"\n[步骤1] 亮像素质心 (建议圆心): cx={cx:.1f}, cy={cy:.1f}")

    # === 步骤 2: 检测半径 ===
    r, radii, means = find_container_radius(arr, cx, cy)
    print(f"[步骤2] 亮度断崖检测 (建议半径): r={r} px")

    # === 步骤 3: 输出建议参数 ===
    print(f"\n{'='*50}")
    print(f"  建议参数: cx={cx:.0f}  cy={cy:.0f}  r={r}")
    print(f"  (用于后续波长提取脚本)")
    print(f"{'='*50}")

    # === 步骤 4: 生成标注图 ===
    output_dir = os.path.dirname(img_path) if os.path.dirname(img_path) else '.'
    output_path = os.path.join(output_dir, "container_detected.png")
    annotate_and_save(img_path, cx, cy, r, output_path)

    print(f"\n请打开 container_detected.png 验证：")
    print(f"  - 绿圈是否贴合容器边缘？")
    print(f"  - 灰圈 (±5px) 哪个更准？")
    print(f"  - 圆心十字是否在液面中心白点上？")
    print(f"  如需微调，直接在脚本中修改 cx, cy, r 即可。")


if __name__ == '__main__':
    main()
