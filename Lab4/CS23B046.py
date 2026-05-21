import sys
import time
import numpy as np
from scipy.sparse import csr_matrix

# ── Constants ────────────────────────────────────────────────────────────
Q            = 3329        # Kyber modulus
MONT_FACTOR  = 767         # Montgomery correction: inv(2704) mod 3329
SHIFTS       = list(range(-5, 6))   # 11 shifts around detected peak


# ── Data loading ─────────────────────────────────────────────────────────
def load_data(poly_file, trace_file):
    """Load polynomial coefficient CSV and power trace CSV."""
    print(f"Loading {poly_file} ...", flush=True)
    t0 = time.time()
    polys = np.loadtxt(poly_file, delimiter=',', dtype=np.int32)
    print(f"    shape={polys.shape}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"Loading {trace_file} ...", flush=True)
    t0 = time.time()
    traces = np.loadtxt(trace_file, delimiter=',', dtype=np.float32)
    print(f"    shape={traces.shape}  ({time.time()-t0:.1f}s)", flush=True)
    return polys, traces


# ── Step 1: Auto-detect leakage time samples ─────────────────────────────
def detect_peaks(polys, traces):
    N, T = traces.shape
    C    = polys.shape[1]

    # Normalise traces column-wise (zero-mean, unit std per sample)
    tr_c  = (traces - traces.mean(axis=0)).astype(np.float32)
    tr_cn = (tr_c / (tr_c.std(axis=0) + 1e-12)).astype(np.float32)

    row_idx = np.arange(N, dtype=np.int32)
    peaks   = np.zeros(C, dtype=np.int32)

    for ci in range(C):
        # Build sparse indicator matrix: row n fires at column polys[n, ci]
        sp       = csr_matrix(
                       (np.ones(N, dtype=np.float32), (row_idx, polys[:, ci])),
                       shape=(N, Q))
        W_ci     = sp.T @ tr_cn              # (Q, T)  projection
        col_norms = np.linalg.norm(W_ci, axis=0)   # (T,)
        peaks[ci] = int(np.argmax(col_norms))

    return peaks


# ── Step 2: Build per-coefficient summary vectors ─────────────────────────
def build_coeff_tr(polys, traces, peaks, shift=0):

    N, T = traces.shape
    C    = polys.shape[1]

    # Extract one trace sample per coefficient, applying the shift
    tr_mat = np.zeros((N, C), dtype=np.float32)
    for ci in range(C):
        idx = max(0, min(T - 1, int(peaks[ci]) + shift))
        tr_mat[:, ci] = traces[:, idx]

    # Normalise each coefficient's column independently
    tr_c  = tr_mat - tr_mat.mean(axis=0)
    tr_cn = (tr_c / (tr_c.std(axis=0) + 1e-12)).astype(np.float32)

    # Accumulate into (C, Q) summary matrix
    coeff_tr = np.zeros((C, Q), dtype=np.float32)
    for ci in range(C):
        # coeff_tr[ci, r] += tr_cn[n, ci]  for all n where polys[n,ci] == r
        np.add.at(coeff_tr[ci], polys[:, ci], tr_cn[:, ci])

    return coeff_tr


# ── Step 3: CPA via lookup table ──────────────────────────────────────────
def cpa_attack(polys, traces):

    N, C = polys.shape
    T    = traces.shape[1]

    # Hamming weight lookup table for all residues mod q
    lut = np.array([bin(i).count('1') for i in range(Q)], dtype=np.float32)
    v   = np.arange(Q, dtype=np.int32)

    # ── Step 1: Auto-detect leaking time samples ──────────────────────
    print("Detecting leakage time samples ...", flush=True)
    t0    = time.time()
    peaks = detect_peaks(polys, traces)
    print(f"    Done ({time.time()-t0:.1f}s)  "
          f"peaks[0:8] = {peaks[:8].tolist()}", flush=True)

    # ── Step 2: LUT-CPA over all shifts ───────────────────────────────
    print(f"LUT-CPA: {Q} hypotheses x {len(SHIFTS)} shifts x {C} coefficients",
          flush=True)
    t0 = time.time()

    # Accumulators: best score and corresponding raw key per coefficient
    max_corr_g = np.zeros(C, dtype=np.float32)
    best_s_g   = np.zeros(C, dtype=np.int32)

    for shift in SHIFTS:
        coeff_tr = build_coeff_tr(polys, traces, peaks, shift=shift)

        max_corr = np.zeros(C, dtype=np.float32)
        best_s   = np.zeros(C, dtype=np.int32)

        for s in range(Q):
            # lut_perm[r] = HW( r * s mod q )
            lut_perm = lut[np.mod(v * np.int32(s), Q)]
            # score for each coefficient simultaneously: (C,Q).(Q,) -> (C,)
            abs_corr_s = np.abs(coeff_tr @ lut_perm)
            improved   = abs_corr_s > max_corr
            max_corr   = np.where(improved, abs_corr_s, max_corr)
            best_s     = np.where(improved, s,           best_s)

        # Keep global best across all shifts
        improved_g = max_corr > max_corr_g
        max_corr_g = np.where(improved_g, max_corr, max_corr_g)
        best_s_g   = np.where(improved_g, best_s,   best_s_g)

    print(f"CPA done in {time.time()-t0:.1f}s", flush=True)

    # ── Step 3: Montgomery correction ─────────────────────────────────
    # CPA recovers k = s * 2704 mod q  (Montgomery-scaled value).
    # True coefficient: s = k * 767 mod q  (since 2704 * 767 ≡ 1 mod 3329)
    secret = (best_s_g.astype(np.int64) * MONT_FACTOR % Q).astype(np.int32)
    return secret


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 cs23b046.py polynomials.csv power_traces.csv")
        sys.exit(1)

    poly_file, trace_file = sys.argv[1], sys.argv[2]
    t_wall = time.time()

    polys, traces = load_data(poly_file, trace_file)
    secret        = cpa_attack(polys, traces)

    # print(f"[*] S1[0:10] = {secret[:10].tolist()}")
    # print(len(secret))
    out = "CS23B046_key.txt"
    with open(out, 'w') as f:
        f.write('[' + ','.join(map(str, secret)) + ']')
    print(f"Written -> {out}")
    print(f"Total time: {time.time()-t_wall:.1f}s")


if __name__ == "__main__":
    main()