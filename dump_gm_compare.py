"""Compare A/B GM from merged layout quant vs rearrange (split) reference."""
import torch
from pypto.runtime import RunConfig
from tests.st.runtime.ops import test_matmul_mx as t

cfg = RunConfig(platform='a5')
a = t._exact_quantizable_matrix(t.BIG_M, t.BIG_K)
b_nk = t._exact_quantizable_matrix(t.BIG_N, t.BIG_K, transpose_pattern=True)

# --- rearrange path (known good, split) ---
t.quantized_matmul_mx_rearrange_ab_onboard._cache.clear()
# run rearrange only for GM? It only returns out. Need to call quant incore helpers.

a_q_r = torch.empty(t.BIG_M, t.BIG_K, dtype=torch.float8_e4m3fn)
a_s_r = torch.empty(1, t.BIG_G, dtype=torch.float8_e8m0fnu)
b_nk_r = torch.empty(t.BIG_N, t.BIG_K, dtype=torch.float8_e4m3fn)
b_q_r = torch.empty(t.BIG_K, t.BIG_N, dtype=torch.float8_e4m3fn)
b_s_r = torch.empty(1, t.BIG_BG, dtype=torch.float8_e8m0fnu)

t._quantized_matmul_mx_rearrange_quant_a._cache.clear()
t._quantized_matmul_mx_rearrange_quant_b._cache.clear()
a_q_r, a_s_r = t._quantized_matmul_mx_rearrange_quant_a(a, a_q_r, a_s_r, config=cfg)
b_nk_r, b_q_r, b_s_r = t._quantized_matmul_mx_rearrange_quant_b(b_nk, b_nk_r, b_q_r, b_s_r, config=cfg)

# --- layout merged path ---
t._quantized_matmul_mx_layout_quant_ab._cache.clear()
a_q_l = torch.empty_like(a_q_r)
a_s_l = torch.empty_like(a_s_r)
b_nk_l = torch.empty_like(b_nk_r)
b_q_l = torch.empty_like(b_q_r)
b_s_l = torch.empty_like(b_s_r)
a_q_l, a_s_l, b_nk_l, b_q_l, b_s_l = t._quantized_matmul_mx_layout_quant_ab(
    a, b_nk, a_q_l, a_s_l, b_nk_l, b_q_l, b_s_l, config=cfg
)

def cmp(name, x, y):
    xb = x.view(torch.uint8).reshape(-1)
    yb = y.view(torch.uint8).reshape(-1)
    bad = xb != yb
    n = int(bad.sum())
    print(f'{name}: mismatch bytes {n}/{xb.numel()}')
    if n:
        idx = bad.nonzero().flatten()[:16].tolist()
        print('  first bad flat idx', idx)
        if name.startswith('a_q'):
            # show which rows
            rows = sorted({i // t.BIG_K for i in bad.nonzero().flatten().tolist()})
            print('  bad rows', rows[:20], 'count', len(rows))

cmp('a_q', a_q_l, a_q_r)
cmp('a_s', a_s_l, a_s_r)
cmp('b_q', b_q_l, b_q_r)
cmp('b_s', b_s_l, b_s_r)
cmp('b_nk_q', b_nk_l, b_nk_r)
