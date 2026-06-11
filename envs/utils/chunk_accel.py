import numpy as np


def reconstruct_chunk(chunk: np.ndarray, v: float) -> np.ndarray:
    """
    Resample an action chunk at speed v via linear interpolation.

      v > 1  aggregation:     M = floor((N-1)/v)+1 < N  (chunk 变短)
      v < 1  decomposition:   M > N                     (chunk 变长)
      v = 1  identity

    采样位置 positions[k] = k * v, k = 0, 1, ..., M-1, clip 到 [0, N-1].
    对每个浮点位置 p:
        idx  = floor(p);  frac = p - idx
        out  = chunk[idx] + frac * (chunk[idx+1] - chunk[idx])

    Args:
        chunk: (N, D) action chunk, 例如 pi0.5 输出的 50 帧
        v:     正实数速度, 支持任意非整数 (如 1.3)

    Returns:
        new_chunk: (M, D) 重构后的 chunk, 底层仍按帧消费
    """
    assert v > 0, f"v must be positive, got {v}"
    assert chunk.ndim == 2, f"chunk must be (N, D), got shape {chunk.shape}"

    N = chunk.shape[0]
    if N <= 1:
        return chunk.copy()

    M = int(np.floor((N - 1) / v)) + 1
    positions = np.arange(M) * v
    positions = np.clip(positions, 0.0, N - 1)

    idx = np.floor(positions).astype(int)
    frac = positions - idx
    idx_next = np.minimum(idx + 1, N - 1)

    return chunk[idx] + frac[:, None] * (chunk[idx_next] - chunk[idx])
