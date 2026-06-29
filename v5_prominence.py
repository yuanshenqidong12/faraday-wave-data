"""
V5-prominence: Zero-padded FFT (441->2048) with prominence-based peak selection.
Pure signal metric (peak_prominences) — no physics prior in peak selection.
Solves FFT bin discretization — correctly resolves 35Hz from 30Hz.
Companion to v5_improved.py (physics scoring) for cross-validation.

Usage:
  1. Edit EXPERIMENTS list below with your experiment name, cx, cy.
  2. Edit container_dir name if different from '圆形容器_10mm'.
  3. Run: python v5_prominence.py
  4. Results saved to <实验图片>/<exp>/temporal_circular_results_v5_prominence.json
"""
import os, sys, json, math, time as time_mod
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import argrelextrema, peak_prominences

# ============================================================
# CONFIGURATION — adjust these for each experiment
# ============================================================
SCALE = 30.0 / 235
ZP_SIZE = 2048
RHO_LM, RHO_E, G = 6280.0, 1010.0, 9.81
H_LM, H_E = 0.01, 0.00106

EXPERIMENTS = [
    {'name': '2026.6.4',  'cx': 226, 'cy': 231, 'container_dir': '圆形容器_10mm'},
    {'name': '2026.6.8',  'cx': 274, 'cy': 240, 'container_dir': '圆形容器_10mm'},
    {'name': '2026.6.10', 'cx': 330, 'cy': 245, 'container_dir': '圆形容器_10mm_0.3曝光'},
]
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '实验图片')
DIAG_DIR = os.path.join(os.path.dirname(__file__), '..', '实验结果汇总')


def analyze_fft_zp(hp_patch, sq, scale, freq_drive):
    """Zero-padded FFT with prominence-based peak selection (no physics prior)."""
    freq_f = freq_drive / 2.0
    omega = 2 * math.pi * freq_f

    wy, wx = np.hanning(sq), np.hanning(sq)
    patch_win = hp_patch * np.outer(wy, wx)

    zp = np.zeros((ZP_SIZE, ZP_SIZE), dtype=np.float64)
    offset = (ZP_SIZE - sq) // 2
    zp[offset:offset+sq, offset:offset+sq] = patch_win

    fft = np.fft.fft2(zp)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift)**2
    center = ZP_SIZE // 2

    max_r = ZP_SIZE // 2
    yg, xg = np.meshgrid(np.arange(ZP_SIZE), np.arange(ZP_SIZE), indexing='ij')
    dist_arr = np.sqrt((xg - center)**2 + (yg - center)**2).astype(np.int32)
    dist_arr = np.clip(dist_arr, 0, max_r - 1)

    radial_power = np.bincount(dist_arr.ravel(), weights=power.ravel(), minlength=max_r)
    counts = np.bincount(dist_arr.ravel(), minlength=max_r)
    radial_power = radial_power / np.maximum(counts, 1)

    r_lo = max(3, int(ZP_SIZE * scale * 2 / 18))
    r_hi = min(max_r - 1, int(ZP_SIZE * scale * 2 / 5))
    segment = radial_power[r_lo:r_hi]
    local_max = argrelextrema(segment, np.greater, order=4)[0]

    if len(local_max) == 0:
        return {'error': 'no peaks found'}

    prom = peak_prominences(segment, local_max)[0]

    candidates = []
    for i, idx in enumerate(local_max):
        r_px = r_lo + idx
        if r_px <= 5:
            continue

        power_val = segment[idx]
        lo_bg = max(0, idx - 8)
        hi_bg = min(len(segment), idx + 9)
        nearby = np.delete(segment[lo_bg:hi_bg], idx - lo_bg)
        local_bg = np.median(nearby) if len(nearby) > 0 else 1e-15
        snr = power_val / max(local_bg, 1e-15)

        lam_mm = ZP_SIZE * scale / r_px * 2
        k = 2 * math.pi / (lam_mm / 1000.0)
        kh_lm = k * H_LM
        kh_e = k * H_E
        denom = RHO_LM / np.tanh(kh_lm) + RHO_E / np.tanh(kh_e)
        sigma = (omega**2 * denom - (RHO_LM - RHO_E) * G * k) / max(k**3, 1e-6)

        candidates.append({
            'r': r_px, 'lam_true': lam_mm, 'snr': float(snr),
            'prominence': float(prom[i]), 'power': float(power_val), 'sigma': sigma,
        })

    candidates.sort(key=lambda c: c['prominence'], reverse=True)
    best = candidates[0]

    r_int = best['r']
    if r_int > 1 and r_int < max_r - 1:
        y_m1 = radial_power[r_int - 1]
        y_0  = radial_power[r_int]
        y_p1 = radial_power[r_int + 1]
        denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        if abs(denom) > 1e-15:
            delta = (y_m1 - y_p1) / denom
            delta = max(-0.5, min(0.5, delta))
        else:
            delta = 0.0
    else:
        delta = 0.0

    r_refined = float(r_int) + delta
    lam_true_r = ZP_SIZE * scale / r_refined * 2
    k_r = 2 * math.pi / (lam_true_r / 1000.0)
    kh_lm_r = k_r * H_LM
    kh_e_r = k_r * H_E
    denom_r = RHO_LM / np.tanh(kh_lm_r) + RHO_E / np.tanh(kh_e_r)
    sigma_r = (omega**2 * denom_r - (RHO_LM - RHO_E) * G * k_r) / max(k_r**3, 1e-6)

    quality = 'strong' if best['snr'] > 30 else ('reliable' if best['snr'] > 1.5 else ('weak' if best['snr'] > 0.8 else 'noise'))

    return {
        'lam_true': round(lam_true_r, 3),
        'r_int': best['r'], 'r_refined': round(r_refined, 4),
        'delta_bin': round(delta, 4),
        'snr': round(best['snr'], 1), 'sigma': round(sigma_r, 6),
        'quality': quality,
        'n_candidates': len(candidates),
    }


def analyze_condition(img_dir, freq_drive, cx, cy, r_full=220, r_mask=160):
    """Process one experimental condition. Returns dict with analysis results
    plus '_time_avg' and '_bg_subtracted' image arrays for diagnostics."""
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    n_frames = len(files)
    fp0 = os.path.join(img_dir, files[0])
    img0 = Image.open(fp0).convert('L')
    h_img, w_img = img0.size[1], img0.size[0]

    x1 = max(0, cx - r_full)
    x2 = min(w_img, cx + r_full + 1)
    y1 = max(0, cy - r_full)
    y2 = min(h_img, cy + r_full + 1)

    yg, xg = np.ogrid[:y2-y1, :x2-x1]
    dist = np.sqrt((xg - (cx - x1))**2 + (yg - (cy - y1))**2)
    mask = dist <= r_mask

    sum_img = np.zeros((y2-y1, x2-x1), dtype=np.float64)
    for fname in files[:n_frames]:
        roi = np.array(Image.open(os.path.join(img_dir, fname)).convert('L'),
                      dtype=np.float64)[y1:y2, x1:x2]
        sum_img += roi
    time_avg = sum_img / n_frames

    sq = (x2 - x1)
    tam = float(time_avg[mask].mean())

    radial_p = np.zeros(sq//2)
    cnts = np.zeros(sq//2)
    for yi in range(y2-y1):
        for xi in range(x2-x1):
            r = int(dist[yi, xi])
            if r < sq//2:
                radial_p[r] += time_avg[yi, xi]
                cnts[r] += 1
    radial_p = radial_p / np.maximum(cnts, 1)
    valid = cnts > 10
    coeffs = np.polyfit(np.arange(sq//2)[valid], radial_p[valid], 4)
    radial_fit = np.polyval(coeffs, dist.flatten()).reshape(dist.shape)
    time_avg_flat = time_avg - radial_fit
    time_avg_flat[~mask] = 0

    blurred = ndimage.gaussian_filter(time_avg_flat, sigma=20)
    hp = time_avg_flat - blurred
    hp[~mask] = 0

    r = analyze_fft_zp(hp, sq, SCALE, freq_drive)
    r['time_avg_mean'] = tam
    r['_time_avg'] = time_avg
    r['_bg_subtracted'] = time_avg_flat
    return r


def save_diag_image(arr, freq, suffix):
    """Save a diagnostic image to 实验结果汇总/<freq>Hz/<suffix>_10mm.png"""
    freq_dir = os.path.join(DIAG_DIR, f'{freq}Hz')
    os.makedirs(freq_dir, exist_ok=True)
    arr_norm = arr - arr.min()
    if arr_norm.max() > 0:
        arr_norm = (arr_norm / arr_norm.max() * 255).astype(np.uint8)
    else:
        arr_norm = arr.astype(np.uint8)
    Image.fromarray(arr_norm).save(os.path.join(freq_dir, f'{suffix}_10mm.png'))


def process_exp(exp_cfg):
    base = os.path.join(BASE_DIR, exp_cfg['name'])
    cx, cy = exp_cfg['cx'], exp_cfg['cy']
    container_dir = exp_cfg.get('container_dir', '圆形容器_10mm')

    conditions = []
    cpath = os.path.join(base, container_dir)
    if not os.path.isdir(cpath):
        print(f"ERROR: container directory not found: {cpath}")
        return []
    for freq_d in sorted(os.listdir(cpath)):
        fpath = os.path.join(cpath, freq_d)
        if not os.path.isdir(fpath): continue
        try: freq = float(freq_d.replace('Hz','').replace('hz',''))
        except: continue
        for cond in sorted(os.listdir(fpath)):
            cp = os.path.join(fpath, cond)
            if not os.path.isdir(cp): continue
            if not [f for f in os.listdir(cp) if f.lower().endswith('.jpg')]: continue
            conditions.append({'path': os.path.join(container_dir, freq_d, cond), 'freq': freq})

    print(f"\n{'='*85}")
    print(f"V5-PROMINENCE: {exp_cfg['name']} | {len(conditions)} cond | ZP={ZP_SIZE}px")
    print(f"{'='*85}")

    results = []
    saved_freqs = set()
    for i, cond in enumerate(conditions):
        fp = os.path.join(base, cond['path'])
        short = cond['path']
        t0 = time_mod.time()
        r = analyze_condition(fp, cond['freq'], cx, cy)
        et = time_mod.time() - t0

        if 'error' in r:
            print(f"[{i+1:2d}/{len(conditions)}] {short:<45s} ERR: {r['error']}")
            results.append({'path': cond['path'], 'freq': cond['freq'], 'error': r['error']})
        else:
            marker = ' **' if r['snr'] >= 3 and 0.4 < r['sigma'] < 1.3 else ''
            print(f"[{i+1:2d}/{len(conditions)}] {short:<45s} r={r['r_int']:>4d} d={r['delta_bin']:+.3f} "
                  f"lam={r['lam_true']:>7.3f} sig={r['sigma']:.4f} SNR={r['snr']:.1f} {r['quality']}{marker} ({et:.1f}s)")
            results.append({'path': cond['path'], 'freq': cond['freq'],
                           'lam_true': r['lam_true'], 'sigma': r['sigma'],
                           'snr': r['snr'], 'quality': r['quality'],
                           'r_int': r['r_int'], 'r_refined': r['r_refined'],
                           'delta_bin': r['delta_bin'],
                           'time_avg_mean': r['time_avg_mean']})

            if cond['freq'] not in saved_freqs and r.get('snr', 0) >= 1.5:
                freq_int = int(cond['freq'])
                save_diag_image(r['_time_avg'], freq_int, 'time_avg')
                save_diag_image(r['_bg_subtracted'], freq_int, 'bg_subtracted')
                saved_freqs.add(cond['freq'])

        if '_time_avg' in r: del r['_time_avg']
        if '_bg_subtracted' in r: del r['_bg_subtracted']

    ok = [r for r in results if 'error' not in r and r.get('snr',0) >= 2 and 0.3 < r.get('sigma',0) < 2.0]
    good = [r for r in ok if r.get('snr',0) >= 3 and 0.4 < r.get('sigma',0) < 1.3]
    print(f"\n  OK: {len(ok)}/{len(results)}  |  GOOD: {len(good)}")

    jp = os.path.join(base, 'temporal_circular_results_v5_prominence.json')
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEnc)
    print(f"  Saved: {jp}")
    return results


def main():
    for exp in EXPERIMENTS:
        process_exp(exp)
    print(f"\n{'='*85}\nV5-PROMINENCE ALL DONE\n{'='*85}")


if __name__ == '__main__':
    main()
