"""JAX/GPU backend for the position-corrected patch stitching.

This mirrors :func:`ptyrannosaur.engine.stitching.masked_correlation_patch_stitching`
but runs the array-heavy phases (the FFT cross-correlations, peak finding, the
upsampled-DFT shift refinement and the final windowed patch shift/accumulate) in
JAX so they execute on the GPU. The two inherently sequential / graph pieces --
building and pruning the shift-vector graph (``networkx``) and reconciling the
patch positions (a small sparse linear solve) -- stay on the CPU and are reused
unchanged from the numpy module.

Selected via ``learn_stitch(..., backend='jax')``. The numpy backend remains the
default and is bitwise unchanged. The JAX backend runs in float32 (GPUs are slow
at float64); this matches the float64 numpy result to ~1e-7 relative on the
stitched image, i.e. essentially identically.
"""

import numpy as np
from scipy import signal
import jax
import jax.numpy as jnp

from .stitching import (
    build_shift_vector_graph,
    reconcile_shift_vectors,
    ShiftVectors,
    NoShiftVectors,
    StitchResult,
    StitchedPatches,
)


def _find_peaks_jax(cross_corr_abs, expected_shift, penalty_factor, im_shape):
    """JAX port of ``find_peaks_in_cross_corr_abs`` (peak = local max / penalty)."""
    ny, nx = cross_corr_abs.shape[2], cross_corr_abs.shape[3]
    # local maximum over the 8 periodic neighbours
    is_peak = jnp.ones(cross_corr_abs.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr != 0 or dc != 0:
                is_peak &= cross_corr_abs >= jnp.roll(cross_corr_abs, (dr, dc), axis=(2, 3))

    xarr, yarr = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny))
    y_exp, x_exp = expected_shift
    dy = jnp.minimum(jnp.minimum(jnp.abs(yarr - y_exp), jnp.abs(yarr - y_exp - ny)),
                     jnp.abs(yarr - y_exp + ny))
    dx = jnp.minimum(jnp.minimum(jnp.abs(xarr - x_exp), jnp.abs(xarr - x_exp - nx)),
                     jnp.abs(xarr - x_exp + nx))
    dr = jnp.sqrt(dx ** 2 + dy ** 2)
    penalty = dr * penalty_factor + 1
    arr = cross_corr_abs * is_peak / penalty[None, None, :, :]

    flat = jnp.argmax(jnp.reshape(jnp.abs(arr), (arr.shape[0], arr.shape[1], -1)), axis=2)
    maxima = jnp.unravel_index(flat, tuple(int(s) for s in im_shape))
    midpoint = jnp.trunc(jnp.asarray(im_shape, dtype=arr.dtype) / 2)
    shift_n = jnp.stack(maxima).astype(arr.dtype)
    shift_n = jnp.where(shift_n > midpoint[:, None, None], shift_n - im_shape[:, None, None], shift_n)
    return shift_n


def _refine_shift_batch_jax(shifts, image_products, uf, ups, dftshift):
    """JAX port of ``refine_shift_batch`` (batched upsampled-DFT refinement)."""
    M = image_products.shape[0]
    if M == 0:
        return jnp.zeros((2, 0))
    shifts = jnp.round(shifts * uf) / uf
    offsets = dftshift - shifts * uf                 # (2, M): [rows, cols]
    data = jnp.conj(image_products)                  # (M, R, C)
    _, R, C = data.shape
    ups = int(ups)
    ar = jnp.arange(ups)
    cdtype = data.dtype

    freq_c = jnp.fft.fftfreq(C, uf)
    base_c = jnp.exp(-1j * 2 * jnp.pi * ar[:, None] * freq_c[None, :]).astype(cdtype)
    phase_c = jnp.exp(1j * 2 * jnp.pi * offsets[1][:, None] * freq_c[None, :]).astype(cdtype)
    data_c = data * phase_c[:, None, :]
    tmp = jnp.einsum('uc,mrc->mur', base_c, data_c)

    freq_r = jnp.fft.fftfreq(R, uf)
    base_r = jnp.exp(-1j * 2 * jnp.pi * ar[:, None] * freq_r[None, :]).astype(cdtype)
    phase_r = jnp.exp(1j * 2 * jnp.pi * offsets[0][:, None] * freq_r[None, :]).astype(cdtype)
    tmp = tmp * phase_r[:, None, :]
    cc = jnp.conj(jnp.einsum('xr,mur->mxu', base_r, tmp))

    flat = jnp.argmax(jnp.abs(cc).reshape(M, -1), axis=1)
    mr, mc = jnp.unravel_index(flat, (ups, ups))
    maxima = jnp.stack([mr, mc]).astype(cc.real.dtype) - dftshift
    return shifts + maxima / uf


def _batch_shift_bilinear_jax(imgs, shifts):
    """JAX bilinear shift matching ``ndi.shift(order=1, mode='constant', cval=0)``."""
    imgs = jnp.asarray(imgs)
    single = imgs.ndim == 2
    H, W = imgs.shape[-2:]
    n = shifts.shape[0]
    sy = shifts[:, 0].reshape(n, 1, 1)
    sx = shifts[:, 1].reshape(n, 1, 1)
    oy = jnp.arange(H).reshape(1, H, 1)
    ox = jnp.arange(W).reshape(1, 1, W)
    ys = oy - sy
    xs = ox - sx
    valid = (ys >= 0) & (ys <= H - 1) & (xs >= 0) & (xs <= W - 1)
    ysc = jnp.clip(ys, 0, H - 1)
    xsc = jnp.clip(xs, 0, W - 1)
    y0 = jnp.floor(ysc).astype(jnp.int32); fy = ysc - y0
    x0 = jnp.floor(xsc).astype(jnp.int32); fx = xsc - x0
    y1 = jnp.minimum(y0 + 1, H - 1); x1 = jnp.minimum(x0 + 1, W - 1)
    y0b, y1b = jnp.broadcast_to(y0, (n, H, W)), jnp.broadcast_to(y1, (n, H, W))
    x0b, x1b = jnp.broadcast_to(x0, (n, H, W)), jnp.broadcast_to(x1, (n, H, W))
    fyb, fxb = jnp.broadcast_to(fy, (n, H, W)), jnp.broadcast_to(fx, (n, H, W))
    if single:
        g = lambda yy, xx: imgs[yy, xx]
    else:
        nidx = jnp.arange(n).reshape(n, 1, 1)
        g = lambda yy, xx: imgs[nidx, yy, xx]
    out = ((1 - fyb) * (1 - fxb) * g(y0b, x0b) + (1 - fyb) * fxb * g(y0b, x1b)
           + fyb * (1 - fxb) * g(y1b, x0b) + fyb * fxb * g(y1b, x1b))
    return jnp.where(valid, out, 0.0)


def _stitch_patches_jax(data, r_patch, c_patch, patch_pos):
    """JAX port of ``stitch_patches`` (windowed subpixel shift + scatter-add)."""
    padded_data = np.pad(data, ((0, 0), (0, 0), (1, 1), (1, 1)))
    im_shape = data.shape[2:]
    padded_support = np.pad(np.ones(im_shape, dtype=float), 1)
    padded_shape = np.array(padded_support.shape)

    window = signal.windows.hamming(padded_data.shape[2])
    window2 = window[:, None] * window[None, :]
    windowed_padded_support = padded_support * window2
    windowed_padded_data = padded_data * window2[None, None, :, :]

    patch_pos -= np.min(patch_pos, axis=1)[:, np.newaxis]   # in-place (returned)
    region_max = np.ceil(np.max(patch_pos, axis=1)).astype(int) + padded_shape
    padded_pos = patch_pos + 1
    ps0, ps1 = int(padded_shape[0]), int(padded_shape[1])

    rounded_pos = np.round(patch_pos)
    subpx = jnp.asarray((patch_pos - rounded_pos).T)          # (npos, 2)

    gathered = jnp.asarray(windowed_padded_data[r_patch, c_patch])   # (npos, ps0, ps1)
    shifted_patches = _batch_shift_bilinear_jax(gathered, subpx)
    shifted_support = _batch_shift_bilinear_jax(jnp.asarray(windowed_padded_support), subpx)

    npos = len(r_patch)
    r0 = jnp.asarray(rounded_pos[0].astype(np.int32))
    c0 = jnp.asarray(rounded_pos[1].astype(np.int32))
    ay = jnp.arange(ps0); ax = jnp.arange(ps1)
    iy = r0[:, None, None] + ay[None, :, None]
    ix = c0[:, None, None] + ax[None, None, :]
    ncols = int(region_max[1])
    flat = (jnp.broadcast_to(iy, (npos, ps0, ps1)) * ncols
            + jnp.broadcast_to(ix, (npos, ps0, ps1))).reshape(-1)
    size = int(region_max[0]) * ncols
    canvas = jnp.zeros(size).at[flat].add(shifted_patches.reshape(-1)).reshape(tuple(region_max))
    canvas_support = jnp.zeros(size).at[flat].add(shifted_support.reshape(-1)).reshape(tuple(region_max))

    return StitchedPatches(np.asarray(canvas), np.asarray(canvas_support),
                           padded_pos, r_patch, c_patch)


def masked_correlation_patch_stitching_jax(
        data, shift_xn_guess, shift_yn_guess,
        correlation_peak_penalty_factor=2.0,
        shift_vector_mismatch_cutoff=2.2,
        upsample_factor=10,
        no_shift_weight=0.3,
        diagonal_weight=0.5,
        learning_rates=None,
        support_threshold=0.1):
    """JAX/GPU version of ``masked_correlation_patch_stitching``.

    Same signature and return type; see that function for the parameter meanings.
    """
    nr, nc = data.shape[:2]
    shift_dn_guess = shift_yn_guess + shift_xn_guess
    shift_an_guess = shift_yn_guess - shift_xn_guess

    # ---- Phase A: cross-correlations + peak finding (on device) --------------
    P = data.shape[2]
    window = jnp.asarray(signal.windows.hamming(P))
    data_dev = jnp.asarray(np.asarray(data), dtype=jnp.float32)
    windowed = data_dev * window[None, None, :, None] * window[None, None, None, :]
    data_freq = jnp.fft.fftn(windowed, axes=(2, 3))
    im_shape = jnp.asarray(data_freq.shape[2:])

    prod_yn = data_freq[:-1, :] * jnp.conj(data_freq[1:, :])
    prod_xn = data_freq[:, :-1] * jnp.conj(data_freq[:, 1:])
    prod_dn = data_freq[:-1, :-1] * jnp.conj(data_freq[1:, 1:])
    prod_an = data_freq[:-1, 1:] * jnp.conj(data_freq[1:, :-1])

    eps = float(np.finfo(np.float32).eps)
    prod_yn = prod_yn / jnp.maximum(jnp.abs(prod_yn), 100 * eps)
    prod_xn = prod_xn / jnp.maximum(jnp.abs(prod_xn), 100 * eps)
    prod_dn = prod_dn / jnp.maximum(jnp.abs(prod_dn), 100 * eps)
    prod_an = prod_an / jnp.maximum(jnp.abs(prod_an), 100 * eps)

    cc_y = jnp.abs(jnp.fft.ifftn(prod_yn, axes=(2, 3)))
    cc_x = jnp.abs(jnp.fft.ifftn(prod_xn, axes=(2, 3)))
    cc_d = jnp.abs(jnp.fft.ifftn(prod_dn, axes=(2, 3)))
    cc_a = jnp.abs(jnp.fft.ifftn(prod_an, axes=(2, 3)))

    pf = correlation_peak_penalty_factor
    shift_yn = _find_peaks_jax(cc_y, shift_yn_guess, pf, im_shape)
    shift_xn = _find_peaks_jax(cc_x, shift_xn_guess, pf, im_shape)
    shift_dn = _find_peaks_jax(cc_d, shift_dn_guess, pf, im_shape)
    shift_an = _find_peaks_jax(cc_a, shift_an_guess, pf, im_shape)

    # small arrays -> host for the graph / reconcile steps
    shift_yn = np.asarray(shift_yn); shift_xn = np.asarray(shift_xn)
    shift_dn = np.asarray(shift_dn); shift_an = np.asarray(shift_an)

    mismatch_yn = np.linalg.norm(shift_yn - shift_yn_guess[:, None, None], axis=0)
    mismatch_xn = np.linalg.norm(shift_xn - shift_xn_guess[:, None, None], axis=0)
    mismatch_dn = np.linalg.norm(shift_dn - shift_dn_guess[:, None, None], axis=0)
    mismatch_an = np.linalg.norm(shift_an - shift_an_guess[:, None, None], axis=0)

    connect_yn = mismatch_yn < shift_vector_mismatch_cutoff
    connect_xn = mismatch_xn < shift_vector_mismatch_cutoff
    connect_dn = mismatch_dn < shift_vector_mismatch_cutoff
    connect_an = mismatch_an < shift_vector_mismatch_cutoff

    # ---- Phase B: graph build / prune (CPU, networkx) ------------------------
    region_id_map, connect_yn, connect_xn, connect_dn, connect_an, \
        component_sizes = build_shift_vector_graph(
            connect_yn, connect_xn, connect_dn, connect_an, nr, nc)

    # ---- Phase C: refine shifts (device) + reconcile positions (CPU) ---------
    uf = np.float32(upsample_factor)
    ups = float(np.ceil(uf * 1.5))
    dftshift = float(np.trunc(ups / 2.0))

    def get_region_patch_pos(region_id):
        region_defn = region_id_map == region_id
        same_xn = (region_id_map[:, :-1] == region_id) & (region_id_map[:, 1:] == region_id)
        same_yn = (region_id_map[:-1, :] == region_id) & (region_id_map[1:, :] == region_id)
        same_dn = (region_id_map[:-1, :-1] == region_id) & (region_id_map[1:, 1:] == region_id)
        same_an = (region_id_map[:-1, 1:] == region_id) & (region_id_map[1:, :-1] == region_id)
        rc_xn = connect_xn & same_xn
        rc_yn = connect_yn & same_yn
        rc_dn = connect_dn & same_dn
        rc_an = connect_an & same_an
        rd_xn = np.logical_not(connect_xn) & same_xn
        rd_yn = np.logical_not(connect_yn) & same_yn

        all_shifts = {}
        for suffix, connect, shifts, products in zip(
                'xyda',
                [rc_xn, rc_yn, rc_dn, rc_an],
                [shift_xn, shift_yn, shift_dn, shift_an],
                [prod_xn, prod_yn, prod_dn, prod_an]):
            sr, sc = np.nonzero(connect)
            region_shifts = jnp.asarray(shifts[:, sr, sc])
            products_batch = products[np.asarray(sr), np.asarray(sc), :, :]
            refined = _refine_shift_batch_jax(region_shifts, products_batch, uf, ups, dftshift)
            all_shifts[suffix] = ShiftVectors(sr, sc, np.asarray(refined))

        bad_xn = NoShiftVectors(*np.nonzero(rd_xn))
        bad_yn = NoShiftVectors(*np.nonzero(rd_yn))
        r_patch, c_patch, patch_pos, _ = reconcile_shift_vectors(
            region_defn, all_shifts['x'], all_shifts['y'], all_shifts['d'], all_shifts['a'],
            bad_xn, bad_yn, shift_xn_guess, shift_yn_guess,
            no_shift_weight, diagonal_weight, learning_rates)
        return r_patch, c_patch, patch_pos

    r_patch, c_patch, patch_pos = get_region_patch_pos(0)

    # ---- Phase D: windowed shift + accumulate (device) -----------------------
    stitched_region = _stitch_patches_jax(np.asarray(data), r_patch, c_patch, patch_pos)
    stitched_image = stitched_region.masked_image(support_threshold)
    num_patches_excluded = nr * nc - component_sizes[0].item()
    print(f'Image stitched with {num_patches_excluded} patch(es) excluded')

    return StitchResult(stitched_image, stitched_region.image, stitched_region.support,
                        patch_pos, r_patch, c_patch, num_patches_excluded)
