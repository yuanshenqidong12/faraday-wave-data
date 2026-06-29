"""
曝光检查 & 直方图分析脚本
=========================
功能：对单张或一批实验图像做曝光评估，输出容器区域内的亮度统计。
用途：实验前标定曝光参数 — 确保液面区域 mean 落在 80-140，P50 > 50。

使用方法：
  python exposure_check.py <图片路径或文件夹> [--cx CX --cy CY --r R]

  如果不指定容器参数，会自动检测。
"""

import sys, os, math
import numpy as np
from PIL import Image


def analyze_exposure(img_path, cx, cy, r, r_hot=20):
    """分析单张图像容器区域内的曝光质量"""
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape

    # 提取容器圆环区域（排除中心热点）
    vals = []
    for y in range(h):
        for x in range(w):
            d2 = (x-cx)**2 + (y-cy)**2
            if r_hot**2 <= d2 <= r**2:
                vals.append(arr[y, x])

    if not vals:
        return None

    vals = np.array(vals)
    n = len(vals)
    mean = np.mean(vals)
    p50 = np.percentile(vals, 50)
    p75 = np.percentile(vals, 75)
    p90 = np.percentile(vals, 90)
    p95 = np.percentile(vals, 95)
    under30 = np.sum(vals < 30) / n * 100
    over225 = np.sum(vals > 225) / n * 100
    sat255 = np.sum(vals == 255) / n * 100

    # 曝光评级
    if 80 <= mean <= 150 and under30 < 30:
        grade = "✅ 良好"
    elif 50 <= mean < 80 and under30 < 50:
        grade = "⚠️ 偏暗（可用）"
    elif mean > 180 or over225 > 15:
        grade = "⚠️ 过曝"
    elif mean < 30:
        grade = "❌ 严重欠曝"
    else:
        grade = "⚠️ 需调整"

    return {
        'file': os.path.basename(img_path),
        'mean': mean, 'p50': p50, 'p75': p75, 'p90': p90, 'p95': p95,
        'under30': under30, 'over225': over225, 'sat255': sat255,
        'grade': grade, 'n_pixels': n,
    }


def auto_detect_container(img_path):
    """自动检测容器参数"""
    from container_detector import find_container_center, find_container_radius
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    cx, cy = find_container_center(arr)
    r, _, _ = find_container_radius(arr, cx, cy)
    return int(cx), int(cy), r


def main():
    import argparse
    parser = argparse.ArgumentParser(description='曝光检查脚本')
    parser.add_argument('target', help='图片文件或文件夹路径')
    parser.add_argument('--cx', type=float, help='容器圆心 X')
    parser.add_argument('--cy', type=float, help='容器圆心 Y')
    parser.add_argument('--r', type=float, help='容器半径 (px)')
    args = parser.parse_args()

    target = args.target

    # 收集文件
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = sorted([os.path.join(target, f) for f in os.listdir(target)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)

    if not files:
        print("没有图片文件")
        sys.exit(1)

    # 获取容器参数
    if args.cx and args.cy and args.r:
        cx, cy, r = args.cx, args.cy, args.r
    else:
        print("自动检测容器参数...")
        # 选中间帧或曝光最好的一帧
        mid = files[len(files)//2]
        cx, cy, r = auto_detect_container(mid)
        print(f"  自动检测: cx={cx}, cy={cy}, r={r}")
        print(f"  （可用 --cx/--cy/--r 手动指定）\n")

    # 分析
    print(f"{'文件名':<30s} {'Mean':>6s} {'P50':>5s} {'P95':>5s} {'欠曝<30':>7s} {'>225':>6s} {'饱和':>5s}  评级")
    print('-'*90)

    results = []
    for fp in files[::max(1, len(files)//30)]:  # 最多采样30张
        r = analyze_exposure(fp, cx, cy, r)
        if r:
            results.append(r)
            print(f"{r['file']:<30s} {r['mean']:6.1f} {r['p50']:5.0f} {r['p95']:5.0f} {r['under30']:6.1f}% {r['over225']:5.1f}% {r['sat255']:4.1f}%  {r['grade']}")

    if results:
        means = [r['mean'] for r in results]
        print(f"\n汇总: mean 范围 {min(means):.1f} ~ {max(means):.1f}, "
              f"平均 {np.mean(means):.1f}")
        bad = [r for r in results if '❌' in r['grade']]
        if bad:
            print(f"⚠️ {len(bad)}/{len(results)} 张严重欠曝，建议提高 exposure 或增加照明")
        print(f"目标: mean=80~140, P50>50, 欠曝<30%")


if __name__ == '__main__':
    main()
