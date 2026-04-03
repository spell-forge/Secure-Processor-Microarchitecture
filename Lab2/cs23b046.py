import numpy as np
import ctypes, os, sys, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH    = os.path.join(_SCRIPT_DIR, "libpresent.so")
_C_PATH     = os.path.join(_SCRIPT_DIR, "present.c")

def _ensure_lib():
    if not os.path.exists(_SO_PATH):
        if not os.path.exists(_C_PATH):
            with open(_C_PATH, "w") as f:
                f.write(_C_SOURCE)
        ret = subprocess.call(["gcc", "-O3", "-fPIC", "-shared", "-o", _SO_PATH, _C_PATH])
        if ret != 0:
            raise RuntimeError("Failed to compile present.c — is gcc installed?")
    lib = ctypes.CDLL(_SO_PATH)
    lib.bruteforce_keylow.restype  = ctypes.c_int
    lib.bruteforce_keylow.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    return lib

_lib = _ensure_lib()



SBOX = np.array([0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,
                  0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2], dtype=np.uint8)
HW4  = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)



def pearson_corr_batch(L: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """L: (N,16), traces: (N,T) -> (16,T)"""
    N     = traces.shape[0]
    Lc    = L      - L.mean(axis=0, keepdims=True)
    Tc    = traces - traces.mean(axis=0, keepdims=True)
    num   = (Lc.T @ Tc) / N
    l_std = L.std(axis=0)[:, None]
    t_std = traces.std(axis=0)[None, :]
    denom = np.where(l_std * t_std == 0, 1e-12, l_std * t_std)
    return num / denom


def mutual_information_vectorized(leakage: np.ndarray, traces: np.ndarray,
                                   n_bins: int = 9) -> np.ndarray:
    N, T  = traces.shape
    nb    = n_bins - 1
    l_bins = np.clip(
        np.digitize(leakage,
                    np.linspace(leakage.min(), leakage.max()+1e-9, n_bins)) - 1,
        0, nb-1)
    t_min   = traces.min(axis=0)
    t_range = np.where(traces.max(axis=0) > t_min,
                       traces.max(axis=0) - t_min + 1e-9, 1.0)
    t_bins  = np.clip(((traces - t_min) / t_range * nb).astype(np.int32), 0, nb-1)
    joint   = np.zeros((nb, nb, T), dtype=np.float64)
    np.add.at(joint,
              (l_bins[:, None] * np.ones((1,T), dtype=int),
               t_bins, np.arange(T)[None, :]), 1.0)
    joint /= N
    p_l = joint.sum(axis=1, keepdims=True)
    p_t = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(joint > 0, joint / (p_l*p_t+1e-12), 1.0)
        return np.sum(joint * np.log2(ratio+1e-12), axis=(0,1))


def ks_statistic_vectorized(leakage: np.ndarray, traces: np.ndarray) -> np.ndarray:
    med    = np.median(leakage)
    high   = traces[leakage >= med]   # (N_h, T)
    low    = traces[leakage <  med]   # (N_l, T)
    n_h, T = high.shape
    n_l    = low.shape[0]

    if n_l == 0 or n_h == 0:
        return np.zeros(T, dtype=np.float64)

    # Label high=1, low=0, stack -> (N_h+N_l, T)
    labels   = np.concatenate([np.ones(n_h), np.zeros(n_l)])   # (N,)
    combined = np.concatenate([high, low], axis=0)              # (N, T)

    # Sort each column; carry labels along using argsort
    order    = np.argsort(combined, axis=0)                     # (N, T)
    sorted_labels = labels[order]                               # (N, T)

    # Cumulative sum of labels gives count of high values seen so far
    cum_h = np.cumsum(sorted_labels,     axis=0) / n_h          # (N, T)
    cum_l = np.cumsum(1 - sorted_labels, axis=0) / n_l          # (N, T)

    return np.max(np.abs(cum_h - cum_l), axis=0)                # (T,)


def kl_divergence_vectorized(leakage: np.ndarray, traces: np.ndarray,
                              n_bins: int = 9) -> np.ndarray:
    N, T  = traces.shape
    med   = np.median(leakage)
    high  = traces[leakage >= med]   # (N_h, T)
    low   = traces[leakage <  med]   # (N_l, T)

    if len(high) == 0 or len(low) == 0:
        return np.zeros(T, dtype=np.float64)

    nb = n_bins  # number of bins

    # Normalize each column to [0, nb) using global (all-trace) min/max
    t_min   = traces.min(axis=0)                                      # (T,)
    t_range = np.where(traces.max(axis=0) > t_min,
                       traces.max(axis=0) - t_min + 1e-9, 1.0)       # (T,)

    def to_bins(arr):
        return np.clip(((arr - t_min) / t_range * nb).astype(np.int32), 0, nb - 1)

    h_bins = to_bins(high)   # (N_h, T)
    l_bins = to_bins(low)    # (N_l, T)

    # Build (nb, T) histograms for high and low groups
    p = np.zeros((nb, T), dtype=np.float64)
    q = np.zeros((nb, T), dtype=np.float64)
    np.add.at(p, h_bins, 1.0)   # shape broadcasting: h_bins is (N_h, T)
    np.add.at(q, l_bins, 1.0)

    p += 1e-10; q += 1e-10      # Laplace smoothing
    p /= p.sum(axis=0)           # normalise each column
    q /= q.sum(axis=0)

    # Symmetric KL per column: (nb, T) -> (T,)
    kl = 0.5 * (np.sum(p * np.log(p / q), axis=0) +
                 np.sum(q * np.log(q / p), axis=0))
    return kl


def scores_for_metric(L: np.ndarray, t_sub: np.ndarray, metric: str) -> np.ndarray:
    if metric == "pearson":
        return np.max(np.abs(pearson_corr_batch(L, t_sub)), axis=1)
    elif metric == "mi":
        # n_bins=5 is sufficient for ranking and much faster than default 9
        return np.array([float(np.max(mutual_information_vectorized(L[:,k], t_sub, n_bins=5)))
                         for k in range(16)])
    elif metric == "ks":
        return np.array([float(np.max(ks_statistic_vectorized(L[:,k], t_sub)))
                         for k in range(16)])
    elif metric == "kl":
        return np.array([float(np.max(kl_divergence_vectorized(L[:,k], t_sub)))
                         for k in range(16)])
    else:
        raise ValueError(f"Unknown metric: {metric}")



def _nibble(pts: np.ndarray, si: int) -> np.ndarray:
    b = si // 2
    return (pts[:,b] >> 4) & 0xF if si % 2 == 0 else pts[:,b] & 0xF

def hw_leakage_all(pts: np.ndarray, si: int) -> np.ndarray:
    p = _nibble(pts, si)
    return HW4[SBOX[p[:,None] ^ np.arange(16, dtype=np.uint8)[None,:]]].astype(np.float64)

def hd_leakage_all(pts: np.ndarray, si: int) -> np.ndarray:
    p   = _nibble(pts, si)
    pre = p[:,None] ^ np.arange(16, dtype=np.uint8)[None,:]
    return HW4[pre ^ SBOX[pre]].astype(np.float64)



def cpa_attack(pts: np.ndarray, traces: np.ndarray,
               leakage_model: str = "hw", metric: str = "pearson") -> list:
    leak_fn = hw_leakage_all if leakage_model == "hw" else hd_leakage_all
    top2    = []
    for si in range(16):
        L      = leak_fn(pts, si)
        scores = scores_for_metric(L, traces, metric)
        ranked = np.argsort(scores)[::-1]
        best, second = int(ranked[0]), int(ranked[1])
        top2.append((best, second))
        print(f"  Sbox {si:2d}  1st=0x{best:X} ({scores[best]:.4f})  "
              f"2nd=0x{second:X} ({scores[second]:.4f})")
    return top2




def _to_c_arr(data, n):
    arr = (ctypes.c_uint8 * n)()
    for i in range(n): arr[i] = int(data[i])
    return arr

def _check_candidate_c(rk8: list, pt, ct):
    c_rk = _to_c_arr(rk8, 8)
    c_pt = _to_c_arr(pt,  8)
    c_ct = _to_c_arr(ct,  8)
    kl   = _lib.bruteforce_keylow(c_rk, c_pt, c_ct)
    if kl < 0:
        return None
    return bytes(rk8 + [(kl >> 8) & 0xFF, kl & 0xFF])

def nibbles_to_bytes(nibbles: list) -> list:
    return [(nibbles[i] << 4) | nibbles[i+1] for i in range(0, 16, 2)]

def verify_key_candidates(top2: list, pt, ct):
    base  = [p[0] for p in top2]
    cands = [(list(base), "all-1st")]
    for i in range(16):
        c = list(base); c[i] = top2[i][1]
        cands.append((c, f"swap({i})"))
    for i, j in combinations(range(16), 2):
        c = list(base); c[i] = top2[i][1]; c[j] = top2[j][1]
        cands.append((c, f"swap({i},{j})"))

    print(f"  Verifying {len(cands)} candidates via C brute-force ...")
    for idx, (nibbles, label) in enumerate(cands):
        tier = 0 if idx == 0 else (1 if idx <= 16 else 2)
        if idx in (0, 1, 17):
            tname = ["Tier 0 (0 swaps)", "Tier 1 (1 swap)", "Tier 2 (2 swaps)"][tier]
            print(f"  --- {tname} ---")
        rk8    = nibbles_to_bytes(nibbles)
        master = _check_candidate_c(rk8, pt, ct)
        if master is not None:
            print(f"  ✓ Found at candidate {idx} [{label}]")
            print(f"    Round Key : {' '.join(f'{b:02X}' for b in rk8)}")
            return rk8, master

    print("  ✗ No match in 137 candidates.")
    return None, None



def plot_convergence(pts, traces, true_nibble, sbox_index=0,
                     save_path="convergence_plot.png"):
    counts  = np.unique(np.round(
        np.logspace(np.log10(100), np.log10(len(traces)), 30)).astype(int))
    counts  = counts[counts <= len(traces)]
    p       = _nibble(pts, sbox_index)
    guesses = np.arange(16, dtype=np.uint8)
    results = {k: [] for k in range(16)}

    for n in counts:
        L    = HW4[SBOX[p[:n,None] ^ guesses[None,:]]].astype(np.float64)
        peak = np.max(np.abs(pearson_corr_batch(L, traces[:n])), axis=1)
        for kg in range(16):
            results[kg].append(float(peak[kg]))

    plt.figure(figsize=(10, 5))
    for kg in range(16):
        color = 'red' if kg == true_nibble else 'steelblue'
        alpha = 1.0 if kg == true_nibble else 0.3
        lw    = 2   if kg == true_nibble else 0.8
        label = f"Key 0x{kg:X} (correct)" if kg == true_nibble else None
        plt.plot(counts, results[kg], color=color, alpha=alpha, linewidth=lw, label=label)
    wrong = (true_nibble + 1) % 16
    plt.plot(counts, results[wrong], color='gray', linewidth=1.2, linestyle='--',
             label=f"Key 0x{wrong:X} (wrong)")
    plt.xlabel("Number of Traces"); plt.ylabel("Peak |Correlation|")
    plt.title(f"CPA Convergence — S-box position {sbox_index}")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")



def task_5b_comparison(pts, traces):
    print("\n--- Task 5(b): HW vs HD (Average Peak Correlation) ---")
    hw_peaks, hd_peaks = [], []
    for si in range(16):
        hw_corr = float(np.max(np.abs(pearson_corr_batch(hw_leakage_all(pts, si), traces))))
        hd_corr = float(np.max(np.abs(pearson_corr_batch(hd_leakage_all(pts, si), traces))))
        hw_peaks.append(hw_corr)
        hd_peaks.append(hd_corr)
        print(f"  Sbox {si:2d}  HW={hw_corr:.4f}  HD={hd_corr:.4f}  "
              f"{'HW' if hw_corr >= hd_corr else 'HD'} wins")

    avg_hw, avg_hd = np.mean(hw_peaks), np.mean(hd_peaks)
    print(f"\n  Average Peak Correlation — HW: {avg_hw:.4f}  HD: {avg_hd:.4f}")
    winner = "Hamming Weight (HW)" if avg_hw > avg_hd else "Hamming Distance (HD)"
    print(f"  Conclusion: {winner} is the better leakage model for this device.")



def task_5c_guessing_entropy(pts, traces, true_nibble, sbox_index=0,
                              n_experiments=20, save_path="guessing_entropy.png"):
    N = len(traces)
    # Stop at 500 traces, 10 evenly spread steps on a log scale
    max_traces = min(500, N)
    trace_counts = np.unique(np.round(
        np.logspace(np.log10(10), np.log10(max_traces), 10)).astype(int))
    trace_counts = trace_counts[trace_counts <= max_traces]

    rng     = np.random.default_rng(42)
    metrics = ["pearson", "mi", "ks", "kl"]
    colors  = {"pearson": "steelblue", "mi": "coral", "ks": "green", "kl": "purple"}

    # Precompute full leakage matrix once — subsets are drawn by index
    L_full = hw_leakage_all(pts, sbox_index)   # (N, 16)

    ge_results = {}

    print(f"\n  {'Traces':>8}  {'PEARSON':>10}  {'MI':>10}  {'KS':>10}  {'KL':>10}")
    print("  " + "-" * 58)

    for m in metrics:
        ge_results[m] = []

    for n in trace_counts:
        # Draw all n_experiments subsets at once — (n_experiments, n) index array
        all_idx = np.array([rng.choice(N, size=n, replace=False)
                             for _ in range(n_experiments)])   # (E, n)

        row = {}
        for m in metrics:
            ranks = []
            for e in range(n_experiments):
                idx    = all_idx[e]
                scores = scores_for_metric(L_full[idx], traces[idx], m)
                rank   = int(np.where(np.argsort(scores)[::-1] == true_nibble)[0][0]) + 1
                ranks.append(rank)
            ge = float(np.mean(ranks))
            ge_results[m].append(ge)
            row[m] = ge
        print(f"  {n:>8}  {row['pearson']:>10.2f}  {row['mi']:>10.2f}  {row['ks']:>10.2f}  {row['kl']:>10.2f}")

    # Plot
    plt.figure(figsize=(9, 5))
    for m in metrics:
        plt.plot(trace_counts, ge_results[m], label=m.upper(),
                 color=colors[m], linewidth=2, marker='o', markersize=3)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.8, label='GE=1 (perfect)')
    plt.xlabel("Number of Traces")
    plt.ylabel("Guessing Entropy (avg rank of correct key)")
    plt.title(f"Guessing Entropy vs Traces — S-box {sbox_index}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"\n  Saved: {save_path}")

    return ge_results



def main():
    if len(sys.argv) < 3:
        print("Usage: python3 final_cpa.py plaintexts.npy traces.npy [ciphertexts.npy]")
        sys.exit(1)

    pts_file = sys.argv[1]
    tr_file  = sys.argv[2]
    ct_file  = sys.argv[3] if len(sys.argv) > 3 else None

    print("Loading data ...")
    pts    = np.load(pts_file)
    traces = np.load(tr_file).astype(np.float32)
    cts    = np.load(ct_file) if ct_file and os.path.exists(ct_file) else None

    print(f"Traces     : {traces.shape}")
    print(f"Plaintexts : {pts.shape}")

    print("\n=== CPA — Hamming Weight + Pearson ===")
    top2_hw = cpa_attack(pts, traces, leakage_model="hw", metric="pearson")

    print("\n=== CPA — Hamming Distance + Pearson ===")
    top2_hd = cpa_attack(pts, traces, leakage_model="hd", metric="pearson")

    best_rk = None
    master  = None

    if cts is not None:
        print("\n=== Verifying candidates (HW model) ===")
        best_rk, master = verify_key_candidates(top2_hw, pts[0], cts[0])
        if best_rk is None:
            print("\n=== Verifying candidates (HD model fallback) ===")
            best_rk, master = verify_key_candidates(top2_hd, pts[0], cts[0])
    else:
        print("\n[!] No ciphertexts — cannot verify. Using best CPA nibbles.")

    if best_rk is None:
        best_rk = nibbles_to_bytes([p[0] for p in top2_hw])
        master  = bytes(best_rk + [0x00, 0x00])

    rk_str = "".join(f"{b:02X}" for b in best_rk)
    mk_str = "".join(f"{b:02X}" for b in master)
    print(f"\n{'='*50}")
    print(f"First Round Key : {rk_str}")
    print(f"Master Key      : {mk_str}")
    print(f"{'='*50}")
    print(f"\n{rk_str} {mk_str}")

    # True nibble for sbox 0 from verified round key
    nibble0 = (best_rk[0] >> 4) & 0xF

    print("\n=== HW vs HD Model Comparison (Q5b) ===")
    try:
        task_5b_comparison(pts, traces)
    except Exception as e:
        print(f"  [!] {e}")

    print("\n=== Convergence Plot (Q5a) ===")
    try:
        plot_convergence(pts, traces, true_nibble=nibble0,
                         sbox_index=0, save_path="convergence_plot.png")
    except Exception as e:
        print(f"  [!] {e}")

    print("\n=== Guessing Entropy vs Traces (Q5c, S-box 0) ===")
    try:
        task_5c_guessing_entropy(pts, traces,
                                  true_nibble=nibble0,
                                  sbox_index=0,
                                  n_experiments=10,
                                  save_path="guessing_entropy.png")
    except Exception as e:
        print(f"  [!] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
