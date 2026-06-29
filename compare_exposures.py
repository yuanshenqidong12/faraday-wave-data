"""
全量对比 6.1 vs 6.3 实验曝光效果
=================================
扫描两个实验的全部条件，统计 ROI 亮度分布，生成对比图。
"""

from PIL import Image
import numpy as np
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def roi_stats(img_path, cx, cy, r, r_hot=20):
    """计算圆形ROI区域内的亮度统计"""
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape
    vals = []
    for y in range(h):
        for x in range(w):
            d2 = (x - cx)**2 + (y - cy)**2
            if r_hot**2 <= d2 <= r**2:
                vals.append(arr[y, x])
    vals = np.array(vals)
    return {
        'mean': vals.mean(), 'p50': np.percentile(vals, 50),
        'p10': np.percentile(vals, 10), 'p90': np.percentile(vals, 90),
        'under30': (vals < 30).sum() / len(vals) * 100,
        'over225': (vals > 225).sum() / len(vals) * 100,
        'sat255': (vals == 255).sum() / len(vals) * 100,
        'n_pixels': len(vals)
    }


def discover_conditions(base, subdir_filter=None):
    """自动发现实验条件文件夹。返回 [(rel_path, freq), ...]"""
    conds = []
    for d in sorted(os.listdir(base)):
        dpath = os.path.join(base, d)
        if not os.path.isdir(dpath):
            continue
        # 排除非频率文件夹
        if d in ['方形容器', '标定图', 'cam0', 'cam1']:
            continue
        # 尝试解析频率
        freq = None
        freq_str = d.replace('hz', '').replace('Hz', '')
        try:
            freq = float(freq_str)
        except ValueError:
            pass

        if freq is not None:
            # 直接子文件夹是条件
            for sub in sorted(os.listdir(dpath)):
                subpath = os.path.join(dpath, sub)
                if sub in ['cam0', 'cam1']:
                    continue
                if os.path.isdir(subpath):
                    conds.append((os.path.join(d, sub), freq))
        else:
            # 可能是一级子目录结构（如 方形容器/30Hz/...）
            for sub1 in sorted(os.listdir(dpath)):
                sub1path = os.path.join(dpath, sub1)
                if not os.path.isdir(sub1path):
                    continue
                freq2 = None
                freq_str2 = sub1.replace('hz', '').replace('Hz', '')
                try:
                    freq2 = float(freq_str2)
                except ValueError:
                    pass
                if freq2 is not None:
                    for sub2 in sorted(os.listdir(sub1path)):
                        sub2path = os.path.join(sub1path, sub2)
                        if sub2 in ['cam0', 'cam1']:
                            continue
                        if os.path.isdir(sub2path):
                            conds.append((os.path.join(d, sub1, sub2), freq2))

    return conds


def scan_conditions(base, conds, cx, cy, r, camera_subdir=None):
    """扫描所有条件，返回亮度统计列表"""
    results = []
    for rel_path, freq in conds:
        d = os.path.join(base, rel_path)
        if camera_subdir:
            d = os.path.join(d, camera_subdir)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if not files:
            continue
        mid = os.path.join(d, files[len(files) // 2])
        try:
            s = roi_stats(mid, cx, cy, r)
            s['freq'] = freq
            s['cond'] = rel_path
            s['n_frames'] = len(files)
            results.append(s)
        except Exception as e:
            pass  # skip problematic images
    return results


def main():
    BASE61 = '2026.6.1（实验）'
    BASE63 = '2026.6.3'

    # 6.1 container params
    CX61, CY61, R61 = 341, 237, 193

    # 6.3 round container params (cam1)
    CX63_R, CY63_R, R63_R = 355, 246, 235

    # 6.3 square container params (cam1) — inscribed circle in rect ROI
    CX63_SQ = (160 + 595) // 2  # 377
    CY63_SQ = (32 + 450) // 2   # 241
    R63_SQ = min(595 - 160, 450 - 32) // 2  # 209

    # ========================================
    # Discover all conditions
    # ========================================
    conds_61 = discover_conditions(BASE61)
    print(f"6.1: found {len(conds_61)} conditions")

    conds_63_round = []
    for d in sorted(os.listdir(BASE63)):
        dpath = os.path.join(BASE63, d)
        if not os.path.isdir(dpath) or d in ['方形容器', '标定图']:
            continue
        freq = None
        try:
            freq = float(d.replace('hz', '').replace('Hz', ''))
        except ValueError:
            pass
        if freq is None:
            continue
        for sub in sorted(os.listdir(dpath)):
            subpath = os.path.join(dpath, sub)
            if sub in ['cam0', 'cam1']:
                continue
            if os.path.isdir(subpath) and os.path.isdir(os.path.join(subpath, 'cam1')):
                conds_63_round.append((os.path.join(d, sub), freq))

    conds_63_sq = []
    sq_dir = os.path.join(BASE63, '方形容器')
    if os.path.isdir(sq_dir):
        for d in sorted(os.listdir(sq_dir)):
            dpath = os.path.join(sq_dir, d)
            if not os.path.isdir(dpath):
                continue
            try:
                freq = float(d.replace('hz', '').replace('Hz', ''))
            except ValueError:
                continue
            for sub in sorted(os.listdir(dpath)):
                subpath = os.path.join(dpath, sub)
                if sub in ['cam0', 'cam1']:
                    continue
                if os.path.isdir(subpath) and os.path.isdir(os.path.join(subpath, 'cam1')):
                    conds_63_sq.append((os.path.join('方形容器', d, sub), freq))

    print(f"6.3 round cam1: {len(conds_63_round)} conditions")
    print(f"6.3 square cam1: {len(conds_63_sq)} conditions")

    # ========================================
    # Scan all conditions
    # ========================================
    print("\nScanning 6.1...")
    all_61 = scan_conditions(BASE61, conds_61, CX61, CY61, R61)
    print(f"  -> {len(all_61)} valid results")

    print("Scanning 6.3 round cam1...")
    all_63r = scan_conditions(BASE63, conds_63_round, CX63_R, CY63_R, R63_R, 'cam1')
    print(f"  -> {len(all_63r)} valid results")

    print("Scanning 6.3 square cam1...")
    all_63sq = scan_conditions(BASE63, conds_63_sq, CX63_SQ, CY63_SQ, R63_SQ, 'cam1')
    print(f"  -> {len(all_63sq)} valid results")

    all_63 = all_63r + all_63sq

    # ========================================
    # Compute statistics
    # ========================================
    means_61 = np.array([x['mean'] for x in all_61])
    means_63 = np.array([x['mean'] for x in all_63])

    print("\n" + "=" * 70)
    print("FULL COMPARISON: ALL CONDITIONS")
    print("=" * 70)

    print(f"\n6.1 (exposure=0.01), {len(all_61)} conditions:")
    print(f"  Mean brightness: {means_61.mean():.1f}")
    print(f"  Range: [{means_61.min():.1f}, {means_61.max():.1f}]")
    print(f"  Median: {np.median(means_61):.1f}")
    print(f"  P10/P90: {np.percentile(means_61,10):.1f} / {np.percentile(means_61,90):.1f}")
    print(f"  Mean>=50: {np.sum(means_61>=50)}/{len(all_61)} ({np.sum(means_61>=50)/len(all_61)*100:.0f}%)")
    print(f"  Mean in [80,140]: {np.sum((means_61>=80)&(means_61<=140))}/{len(all_61)}")

    print(f"\n6.3 (exposure=0.1), {len(all_63)} conditions:")
    print(f"  Mean brightness: {means_63.mean():.1f}")
    print(f"  Range: [{means_63.min():.1f}, {means_63.max():.1f}]")
    print(f"  Median: {np.median(means_63):.1f}")
    print(f"  P10/P90: {np.percentile(means_63,10):.1f} / {np.percentile(means_63,90):.1f}")
    print(f"  Mean>=50: {np.sum(means_63>=50)}/{len(all_63)} ({np.sum(means_63>=50)/len(all_63)*100:.0f}%)")
    print(f"  Mean in [80,140]: {np.sum((means_63>=80)&(means_63<=140))}/{len(all_63)}")

    # By frequency
    print("\n--- Brightness by Frequency ---")
    freqs_61 = sorted(set(x['freq'] for x in all_61))
    freqs_63 = sorted(set(x['freq'] for x in all_63))
    all_freqs = sorted(set(freqs_61 + freqs_63))
    print(f"{'Freq':>5s} {'6.1_N':>6s} {'6.1_Mean':>9s} {'6.3_N':>6s} {'6.3_Mean':>9s} {'Ratio':>7s}")
    print("-" * 48)
    for freq in all_freqs:
        m61 = [x['mean'] for x in all_61 if x['freq'] == freq]
        m63 = [x['mean'] for x in all_63 if x['freq'] == freq]
        n61, n63 = len(m61), len(m63)
        avg61 = f"{np.mean(m61):.1f}" if m61 else "-"
        avg63 = f"{np.mean(m63):.1f}" if m63 else "-"
        ratio = f"{np.mean(m63)/np.mean(m61):.1f}x" if m61 and m63 else "-"
        print(f"{freq:>5.0f} {n61:>6d} {avg61:>9s} {n63:>6d} {avg63:>9s} {ratio:>7s}")

    # Brightness distribution
    print("\n--- Brightness Distribution ---")
    bins = [0, 20, 40, 60, 80, 100, 120, 150, 200, 256]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n61 = sum(1 for x in all_61 if lo <= x['mean'] < hi)
        n63 = sum(1 for x in all_63 if lo <= x['mean'] < hi)
        pct61 = n61 / len(all_61) * 100
        pct63 = n63 / len(all_63) * 100
        bar61 = '#' * max(1, int(pct61))
        bar63 = '#' * max(1, int(pct63))
        print(f"  [{lo:>3d}-{hi:>3d}): 6.1 {n61:>2d} ({pct61:4.1f}%) {bar61}")
        print(f"              6.3 {n63:>2d} ({pct63:4.1f}%) {bar63}")

    # ========================================
    # Generate comparison plots
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram: brightness distribution
    ax = axes[0, 0]
    ax.hist(means_61, bins=20, alpha=0.6, label=f'6.1 (exp=0.01)\nn={len(all_61)}, mean={means_61.mean():.0f}',
            color='red', edgecolor='darkred')
    ax.hist(means_63, bins=20, alpha=0.6, label=f'6.3 (exp=0.1)\nn={len(all_63)}, mean={means_63.mean():.0f}',
            color='blue', edgecolor='darkblue')
    ax.axvline(x=80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target min (80)')
    ax.axvline(x=140, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target max (140)')
    ax.set_xlabel('ROI Mean Brightness')
    ax.set_ylabel('Number of Conditions')
    ax.set_title('Brightness Distribution: ALL Conditions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Brightness vs Frequency scatter
    ax = axes[0, 1]
    freqs_61_pts = [x['freq'] for x in all_61]
    freqs_63_pts = [x['freq'] for x in all_63]
    ax.scatter(freqs_61_pts, means_61, c='red', alpha=0.5, s=30, label='6.1 (exp=0.01)')
    ax.scatter(freqs_63_pts, means_63, c='blue', alpha=0.5, s=30, label='6.3 (exp=0.1)')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Minimum usable')
    ax.set_xlabel('Driving Frequency (Hz)')
    ax.set_ylabel('ROI Mean Brightness')
    ax.set_title('Brightness vs Frequency (all conditions)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 6.3 only: brightness vs gain effect (color by freq)
    ax = axes[1, 0]
    # Extract gain from condition name
    gains_63 = []
    for x in all_63:
        cond = x['cond']
        # Try to extract gain value from folder name: ag(X.X) or ac_ag(X.X)
        import re
        m = re.search(r'ag\((\d+\.?\d*)\)', cond)
        if m:
            gains_63.append(float(m.group(1)))
        else:
            gains_63.append(0)
    gains_63 = np.array(gains_63)
    sc = ax.scatter(gains_63, means_63, c=freqs_63_pts, cmap='viridis', alpha=0.6, s=40)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gain (ag)')
    ax.set_ylabel('ROI Mean Brightness')
    ax.set_title('6.3 (exp=0.1): Brightness vs Gain')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Frequency (Hz)')

    # 4. Box plot: brightness by experiment
    ax = axes[1, 1]
    bp = ax.boxplot([means_61, means_63], labels=['6.1 (exp=0.01)', '6.3 (exp=0.1)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('ROI Mean Brightness')
    ax.set_title('Brightness Distribution Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add stats text
    ax.text(0.5, 0.02,
            f"6.1: median={np.median(means_61):.0f}, >=50: {np.sum(means_61>=50)}/{len(all_61)}\n"
            f"6.3: median={np.median(means_63):.0f}, >=50: {np.sum(means_63>=50)}/{len(all_63)}",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = os.path.join(BASE63, 'exposure_comparison_full.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")

    # Save JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    json_path = os.path.join(BASE63, 'exposure_comparison.json')
    with open(json_path, 'w') as f:
        json.dump({
            '6.1': [{k: convert(v) for k, v in x.items()} for x in all_61],
            '6.3': [{k: convert(v) for k, v in x.items()} for x in all_63]
        }, f)
    print(f"Data saved: {json_path}")


if __name__ == '__main__':
    main()
