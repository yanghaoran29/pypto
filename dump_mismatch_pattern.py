import torch
from pypto.runtime import RunConfig
from tests.st.runtime.ops import test_matmul_mx as t

cfg = RunConfig(platform='a5')
t.quantized_matmul_mx_layout_ab_onboard._cache.clear()
a = t._exact_quantizable_matrix(t.BIG_M, t.BIG_K)
b_nk = t._exact_quantizable_matrix(t.BIG_N, t.BIG_K, transpose_pattern=True)
expected = torch.matmul(a, b_nk.T)
out = torch.empty_like(expected)
out_acc = torch.empty_like(expected)
t.quantized_matmul_mx_layout_ab_onboard(a, b_nk, out, out_acc, config=cfg)
bad = ~torch.isclose(out, expected, rtol=1e-5, atol=1e-3)
print('bad count', int(bad.sum()), '/', bad.numel())
print('bad rows', bad.any(dim=1).nonzero().flatten().tolist())
print('bad cols count', int(bad.any(dim=0).sum()))
for mi in range(0, 32, 16):
    for ni in range(0, 64, 16):
        nbad = int(bad[mi:mi + 16, ni:ni + 16].sum())
        if nbad:
            print(f'block M[{mi}:{mi+16}] N[{ni}:{ni+16}] bad={nbad}')
for r in range(32):
    nbad = int(bad[r].sum())
    if nbad:
        cols = bad[r].nonzero().flatten().tolist()
        print(f'row {r}: {nbad} first_cols={cols[:24]}')
bad2 = ~torch.isclose(out_acc, 2 * expected, rtol=1e-5, atol=1e-3)
print('out_acc bad', int(bad2.sum()))
diff = (out - expected).abs()
idx = int(diff.argmax())
print('max abs', float(diff.max()), 'flat_idx', idx, 'rc', idx // 64, idx % 64)
