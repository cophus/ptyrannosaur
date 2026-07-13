"""Stiching functions for recombining patches into the full object."""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy import ndimage as ndi
from scipy import signal
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
from dataclasses import dataclass
import itertools

def grid_stitch(patches, scan_positions):
    """Stitch patches according to their exact scan positions faster."""
    N, H, W, L = patches.shape
    shift = H // 2
    # Offset positions once
    offset = np.min(scan_positions, axis=0) - shift
    scan_positions = scan_positions - offset
    # Compute output size
    max_pos = np.max(scan_positions, axis=0) + shift
    out_shape = (max_pos[0], max_pos[1], L)
    full_obj = np.zeros(out_shape, dtype=patches.dtype)
    count_obj = np.zeros(out_shape, dtype=patches.dtype)
    for s in range(N):
        x, y = scan_positions[s]
        xs = slice(x-shift, x+shift)
        ys = slice(y-shift, y+shift)
        full_obj[xs, ys] += patches[s]
        count_obj[xs, ys] += 1  # broadcasting instead of allocating count array
    return full_obj / np.maximum(count_obj, 1)

def learn_stitch(patches, scan_positions):
    N, H, W, L = patches.shape
    n_1d = int(np.sqrt(N))
    data = patches.reshape(n_1d, n_1d, H, H)
    pos = scan_positions.reshape(n_1d,n_1d,2)
    nr, nc = n_1d, n_1d
    # make shifts
    shift_xn_init = pos[:, 1:, ::-1] - pos[:, :-1, ::-1]
    shift_yn_init = pos[1:, :, ::-1] - pos[:-1, :, ::-1]
    shift_xn_guess = np.mean(shift_xn_init, axis=(0, 1))
    shift_yn_guess = np.mean(shift_yn_init, axis=(0, 1))
    # construct result
    res = masked_correlation_patch_stitching(data, shift_yn_guess, shift_xn_guess)
    return res.stitched_image, res.support, res.patch_pos

def masked_correlation_patch_stitching(data: np.ndarray, 
        shift_xn_guess: np.ndarray, shift_yn_guess: np.ndarray,
        correlation_peak_penalty_factor: float = 2.0, 
        shift_vector_mismatch_cutoff: float = 2.2,
        upsample_factor: int = 10,
        no_shift_weight: float = 0.3, 
        diagonal_weight: float = 0.5,
        learning_rates: tuple[float, int] | list[tuple[float, int]] = [(3e-1, 1000), (1e-1, 1000), (3e-2, 500)],
        support_threshold: float = 0.1):
    """Stitch image patches using cross correlation with masking.

    Requires an initial guess of the shift vectors between rows and columns of patches.
    Tries to find a set of patch positions that is most consistent with the cross correlation
    between neighboring pairs of patches. Some image patches may be excluded from the final stitched image.

    Parameters
    ----------
    data : ndarray
        nr x nc x p x p array containing the image patches to be stitched.
        There are nr x nc patches each of size p x p.  The 1st and 2nd axes of this array are assumed to be along
        the y and x directions, respectively.
    shift_xn_guess : np.ndarray
        Vector of length 2 in the form [dy, dx]. Initial guess of the shift vector in the x direction (the 1st axis of `data`).
    shift_yn_guess : np.ndarray
        Vector of length 2 in the form [dy, dx]. Initial guess of the shift vector in the y direction (the 2nd axis of `data`).
    correlation_peak_penalty_factor : float, optional
        Controls how much to penalize peaks in the cross-correlation that are farther from the initial guess.
        The greater this value, the greater the penalty. A value of 1 does not penalize at all.
    shift_vector_mismatch_cutoff : float, optional
        This value is a distance in units of pixels. Shift vectors that are more than this distance from the
        initial guess are excluded from the patch position refinement.
    upsample_factor : int, optional
        How much to upsample the cross correlation by when calculating more refined shift vectors.
    no_shift_weight : float, optional
        In the patch position refinement, when a x- or y-shift-vector value from cross correlation is not available,
        the initial guess is used instead. This controls how much such vectors are weighted.
    diagonal_weight : float, optional
        Weight for the shift vector between pairs of diagonally adjacent patches.
    learning_rates : tuple of (float, int) or list of tuple of (float, int), optional
        Contains one tuple or a list of tuples. Each tuple is (lr, niter) where lr is the learning rate for the Adam optimizer
        and niter is the number of iterations.
    support_threshold: float, optional
        Pixels in the final image with total support less than this threshold are masked out.

    Returns
    -------
    result : `CorrelationStitchResult` instance
        This is an object with the following attributes:
        
        stitched_image : ndarray
            Image stitched from the patches.  Pixels not covered by enough patches are set to NaN.
        patch_sum : ndarray
            Array summed from the patches, not normalized by the number of patches covering each pixel.
        support : ndarray
            Weighted total number of patches ("support") covering each pixel. A Hamming window is applied to the patches 
            before summing so the weight of a patch is smaller at the edges and corners of the patch.
        patch_pos : ndarray
            2xN array of optimized patch coordinates where patch_pos[0] corresponds to y-coordinates
        r_patch : ndarray
            Vectors of the row indices of the N patches
        c_patch : ndarray
            Vectors of the column indices of the N patches
        num_patches_excluded : int
            Number of patches not included in the final stitched image.
    
    Notes
    -----
    This stitching code is based on the phase cross correlation approach implemented 
    in scikit-image, but with several modifications to try to mitigate issues that
    arise when trying to stitch patches of limited field of view containing only a few
    atoms each with inconsistent contrast.

    The process is as follows: 1) Apply a Hamming window to all the patches.
    2) Compute the normalized cross-correlation between all pairs of patches
    that are neighbors in the x, y, diagonal, or anti-diagonal direction.
    This step is just like in phase cross correlation. 3) Find all peaks in
    the cross correlation arrays and penalize the peaks based on how far they
    are from what is expected given the initial shift vector guess. 4) For
    each pair of neighboring patches, pick the largest peak after the penalty
    factor and use the peak position as the shift vector between the pair of
    patches. 5) Filter the shift vectors and exclude all shift vectors that
    are more than `shift_vector_mismatch_cutoff` from the position expected
    given the initial shift vector guess. 6) Build a graph from the patches
    and remaining shift vectors and disconnect patches that are not well-
    connected to other patches. 7) Gather the patches and shift vectors of
    the largest connected component of the graph. 8) Refine the shift vectors
    to subpixel precision by upsampling the cross correlation arrays. 9) Use
    cost function optimization to find a set of patch positions that is most
    consistent with the refined shift vectors. 10) Shift the patches
    according to the patch positions. Calculate the final image as a weighted
    sum of these shifted patches.
    """
    
    nr, nc = data.shape[:2]
    shift_dn_guess = shift_yn_guess + shift_xn_guess
    shift_an_guess = shift_yn_guess - shift_xn_guess

    # window the data
    window = signal.windows.hamming(data.shape[2])
    windowed_data = data * window[np.newaxis, np.newaxis, :, np.newaxis]
    windowed_data *= window[np.newaxis, np.newaxis, np.newaxis, :]

    data_freq = fftn(windowed_data, axes=(2, 3), workers=-1)
    im_shape = np.array(data_freq.shape[2:])

    prod_yn = data_freq[:-1, :, :, :] * data_freq[1:, :].conj()
    prod_xn = data_freq[:, :-1, :, :] * data_freq[:, 1:, :, :].conj()
    prod_dn = data_freq[:-1, :-1, :, :] * data_freq[1:, 1:, :, :].conj()
    prod_an = data_freq[:-1, 1:, :, :] * data_freq[1:, :-1, :, :].conj()

    eps = np.finfo(prod_yn.real.dtype).eps

    prod_yn /= np.maximum(np.abs(prod_yn), 100*eps)
    prod_xn /= np.maximum(np.abs(prod_xn), 100*eps)
    prod_dn /= np.maximum(np.abs(prod_dn), 100*eps)
    prod_an /= np.maximum(np.abs(prod_an), 100*eps)

    cross_corr_y = ifftn(prod_yn, axes=(2,3), workers=-1)
    cross_corr_x = ifftn(prod_xn, axes=(2,3), workers=-1)
    cross_corr_d = ifftn(prod_dn, axes=(2,3), workers=-1)
    cross_corr_a = ifftn(prod_an, axes=(2,3), workers=-1)

    float_dtype = prod_yn.real.dtype

    def find_peaks_in_cross_corr_abs(cross_corr_abs, expected_shift=None):
        if expected_shift is not None:
            # A pixel is a peak iff it is >= all 8 periodic neighbors, i.e. it
            # equals the maximum over its 3x3 (wrap-around) window. A single
            # separable maximum filter computes that far faster than eight full
            # array rolls-and-compares, with identical results.
            is_peak = cross_corr_abs == ndi.maximum_filter(
                cross_corr_abs, size=(1, 1, 3, 3), mode='wrap')

            ny = cross_corr_abs.shape[2]
            nx = cross_corr_abs.shape[3]
            xarr, yarr = np.meshgrid(np.arange(nx), np.arange(ny))
            y_exp, x_exp = expected_shift

            # because the cross correlation array is periodic
            dy = np.minimum(np.minimum(np.abs(yarr - y_exp), np.abs(yarr - y_exp - ny)), np.abs(yarr - y_exp + ny))
            dx = np.minimum(np.minimum(np.abs(xarr - x_exp), np.abs(xarr - x_exp - nx)), np.abs(xarr - x_exp + nx))
            dr = np.sqrt(dx**2 + dy**2)

            # penalize peaks that are too far from what is expected
            penalty = dr * correlation_peak_penalty_factor + 1

            arr = cross_corr_abs * is_peak / penalty[np.newaxis, np.newaxis, :, :]
        else:
            arr = cross_corr_abs

        flat_indices = np.argmax(np.reshape(np.abs(arr), (arr.shape[0], arr.shape[1], -1)), axis=2)
        maxima = np.unravel_index(flat_indices, im_shape)
        midpoint = np.array([np.trunc(axis_size / 2) for axis_size in im_shape])
        shift_n = np.stack(maxima).astype(float_dtype, copy=False)
        for d in range(2):
            shift_n[d][shift_n[d] > midpoint[d]] -= im_shape[d]
        return shift_n

    shift_yn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_y), shift_yn_guess)
    shift_xn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_x), shift_xn_guess)
    shift_dn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_d), shift_dn_guess)
    shift_an = find_peaks_in_cross_corr_abs(np.abs(cross_corr_a), shift_an_guess)

    # how far is the computed image shift from what we would expect with perfect probe positions?
    mismatch_yn = np.linalg.norm(shift_yn - shift_yn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_xn = np.linalg.norm(shift_xn - shift_xn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_dn = np.linalg.norm(shift_dn - shift_dn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_an = np.linalg.norm(shift_an - shift_an_guess[:, np.newaxis, np.newaxis], axis=0)

    connect_yn = mismatch_yn < shift_vector_mismatch_cutoff
    connect_xn = mismatch_xn < shift_vector_mismatch_cutoff
    connect_dn = mismatch_dn < shift_vector_mismatch_cutoff
    connect_an = mismatch_an < shift_vector_mismatch_cutoff

    region_id_map, connect_yn, connect_xn, connect_dn, connect_an, \
        component_sizes = build_shift_vector_graph(connect_yn, connect_xn, connect_dn, connect_an, nr, nc)
    
    
    # refine shift vectors and produce a set of patch positions (time-consuming)
    upsample_factor = np.array(upsample_factor, dtype=float_dtype)
    upsampled_region_size = np.ceil(upsample_factor * 1.5)
    dftshift = np.trunc(upsampled_region_size / 2.0) # Center of output array at dftshift + 1

    def get_region_patch_pos(region_id):
        region_defn = region_id_map == region_id

        direction_suffix = 'xyda'
        same_region_xn = (region_id_map[:, :-1]   == region_id) & (region_id_map[:, 1:] == region_id)
        same_region_yn = (region_id_map[:-1, :]   == region_id) & (region_id_map[1:, :] == region_id)
        same_region_dn = (region_id_map[:-1, :-1] == region_id) & (region_id_map[1:, 1:] == region_id)
        same_region_an = (region_id_map[:-1, 1:]  == region_id) & (region_id_map[1:, :-1] == region_id)
        region_connect_xn = connect_xn & same_region_xn
        region_connect_yn = connect_yn & same_region_yn
        region_connect_dn = connect_dn & same_region_dn
        region_connect_an = connect_an & same_region_an

        region_disconnect_xn = np.logical_not(connect_xn) & same_region_xn
        region_disconnect_yn = np.logical_not(connect_yn) & same_region_yn

        all_shifts = {}

        for suffix, connect, shifts, products in zip(direction_suffix,
                    [region_connect_xn, region_connect_yn, region_connect_dn, region_connect_an],
                    [shift_xn, shift_yn, shift_dn, shift_an],
                    [prod_xn, prod_yn, prod_dn, prod_an]):
            sr, sc = np.nonzero(connect)
            region_shifts = shifts[:, sr, sc]
            # Refine all pairs for this direction at once (batched upsampled DFT).
            refined_shifts = refine_shift_batch(
                region_shifts, products[sr, sc, :, :],
                upsample_factor, upsampled_region_size, dftshift)
            all_shifts[suffix] = ShiftVectors(sr, sc, refined_shifts)

        bad_xn_indices = NoShiftVectors(*np.nonzero(region_disconnect_xn))
        bad_yn_indices = NoShiftVectors(*np.nonzero(region_disconnect_yn))

        r_patch, c_patch, patch_pos, _ = reconcile_shift_vectors(region_defn, all_shifts['x'], all_shifts['y'], all_shifts['d'], all_shifts['a'],
                                                            bad_xn_indices, bad_yn_indices,
                                                            shift_xn_guess, shift_yn_guess, 
                                                            no_shift_weight, diagonal_weight, learning_rates)
        return r_patch, c_patch, patch_pos
    
    r_patch, c_patch, patch_pos = get_region_patch_pos(0)

    stitched_region = stitch_patches(data, r_patch, c_patch, patch_pos)
    stitched_image = stitched_region.masked_image(support_threshold)
    num_patches_excluded = nr*nc - component_sizes[0].item()
    print(f'Image stitched with {num_patches_excluded} patch(es) excluded')

    return StitchResult(stitched_image,
        stitched_region.image, stitched_region.support, 
        patch_pos, r_patch, c_patch, num_patches_excluded)


@dataclass
class StitchResult:
    stitched_image: np.ndarray
    patch_sum: np.ndarray
    support: np.ndarray
    patch_pos: np.ndarray
    r_patch: np.ndarray
    c_patch: np.ndarray
    num_patches_excluded: int


def build_shift_vector_graph(connect_yn, connect_xn, connect_dn, connect_an, nr, nc):
    print("proportion of shift vectors within threshold:")
    for dir_str, arr in zip(['xn', 'yn', 'dn', 'an'], [connect_xn, connect_yn, connect_dn, connect_an]):
        prop = np.sum(arr) / np.size(arr)
        print(f'{dir_str}: {prop*100:.2f}%')

    def coord_to_idx(r, c):
        return r*nc + c

    def idx_to_coord(idx):
        return idx // nc, idx % nc

    # build initial graph
    graph = nx.Graph()
    graph.add_nodes_from(range(nr*nc))
    for r in range(nr):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r, c+1)
            if connect_xn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c)
            if connect_yn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c+1)
            if connect_dn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c+1)
            p2 = coord_to_idx(r+1, c)
            if connect_an[r, c]:
                graph.add_edge(p1, p2)


    def straight_and_diagonal_edges(nr, nc):
        for r in range(nr):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r, c+1)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r+1, c)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r+1, c+1)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c+1)
                p2 = coord_to_idx(r+1, c)
                yield r, c, p1, p2

    # try to disconnect some patches that are not connected by enough shift vectors to the rest of the graph
    n_edge_remove = 0
    n_iterations = 0
    for _ in range(8):
        changed = False

        # remove articulation points
        comm_ids = np.full(nr*nc, -1, dtype=int)
        components = list(sorted(nx.biconnected_components(graph), key=len, reverse=True))
        large_id = 124*124
        for k, c in enumerate(components):
            for node in c:
                if comm_ids[node] == -1:
                    comm_ids[node] = k
                else: # articulation points belonging to multiple components shall be removed from both
                    comm_ids[node] = large_id
                    large_id += 1
        for r, c, p1, p2 in straight_and_diagonal_edges(nr, nc):
            if graph.has_edge(p1, p2) and (comm_ids[p1] != comm_ids[p2]):
                graph.remove_edge(p1, p2)
                n_edge_remove += 1
                changed = True

        # remove bridges
        bridge_list = list(nx.bridges(graph))
        if len(bridge_list) > 0:
            n_edge_remove += len(bridge_list)
            graph.remove_edges_from(bridge_list)
            changed = True
        if not changed:
            break
        else:
            n_iterations += 1
    print(f'{n_edge_remove} edge(s) removed over {n_iterations} iteration(s)')
    # print('graph changed after last iteration: ', changed)

    # update_connectivity_arrays
    for r in range(nr):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r, c+1)
            connect_xn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c)
            connect_yn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c+1)
            connect_dn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c+1)
            p2 = coord_to_idx(r+1, c)
            connect_an[r, c] = graph.has_edge(p1, p2)


    components = list(sorted(nx.connected_components(graph), key=len, reverse=True))
    component_sizes = np.array([len(c) for c in components])

    region_ids = np.full(nr*nc, -1, dtype=int)
    components = [np.array(list(c)) for c in components]
    for k, c in enumerate(components):
        region_ids[c] = k
        # print(k, len(c))
    # print(len(components), 'regions', 'min id', np.min(region_ids), 'max id', np.max(region_ids))

    region_id_map = np.reshape(region_ids, (nr, nc))

    return region_id_map, connect_yn, connect_xn, connect_dn, connect_an, component_sizes


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This function was taken from scikit-image:
    https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/registration/_phase_cross_correlation.py

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def refine_shift(shift, image_product, upsample_factor, upsampled_region_size, dftshift):
    '''
    Given a shift vector and the correlation between two image patches, refine the shift to subpixel precision.

    This function was adapted from scikit-image:
    https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/registration/_phase_cross_correlation.py
    '''

    shift = np.round(shift * upsample_factor) / upsample_factor

    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shift * upsample_factor
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()

    # Locate maximum and map back to original pixel grid
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )

    float_dtype = image_product.real.dtype
    maxima = np.stack(maxima).astype(float_dtype, copy=False)
    maxima -= dftshift

    shift += maxima / upsample_factor

    return shift


def refine_shift_batch(shifts, image_products, upsample_factor,
                       upsampled_region_size, dftshift):
    """Vectorized `refine_shift` over many patch pairs at once.

    Produces the same subpixel-refined shift vectors as calling `refine_shift`
    in a loop, but computes the upsampled DFT for all pairs simultaneously via
    batched matrix multiplication (einsum), which is far faster and maps well to
    a GPU.

    Parameters
    ----------
    shifts : ndarray
        2 x M array of integer-pixel shift estimates ([dy, dx] per pair).
    image_products : ndarray
        M x p x p array of the cross-correlation image products for each pair.
    upsample_factor, upsampled_region_size, dftshift
        Same meaning as in `refine_shift` / `_upsampled_dft`.

    Returns
    -------
    ndarray
        2 x M array of subpixel-refined shift vectors.
    """
    uf = upsample_factor
    M = image_products.shape[0]
    if M == 0:
        return np.empty((2, 0), dtype=float)

    shifts = np.round(shifts * uf) / uf              # (2, M)
    # sample_region_offset per pair, matching refine_shift's
    #   sample_region_offset = dftshift - shift * upsample_factor
    offsets = dftshift - shifts * uf                 # (2, M): [rows, cols]

    data = image_products.conj()                     # (M, p, p)
    _, R, C = data.shape
    ups = int(upsampled_region_size)
    ar = np.arange(ups)
    cdtype = data.dtype

    # _upsampled_dft processes axes in reversed order (columns first, then rows).
    # The per-pair kernel factors as a shared base kernel times a per-pair phase:
    #   exp(-2i.pi (u - off_m) f) = exp(-2i.pi u f) * exp(+2i.pi off_m f)
    # so the base kernel (no pair axis) becomes a single shared matmul and only a
    # small (M, size) phase correction is exponentiated per pair. This avoids
    # materializing an (M, ups, size) kernel and maps cleanly onto a GPU GEMM.
    freq_c = fftfreq(C, uf)
    base_c = np.exp(-1j * 2 * np.pi * ar[:, None] * freq_c[None, :]).astype(cdtype)  # (ups, C)
    phase_c = np.exp(1j * 2 * np.pi * offsets[1][:, None] * freq_c[None, :]).astype(cdtype)  # (M, C)
    data_c = data * phase_c[:, None, :]              # (M, R, C)
    tmp = np.einsum('uc,mrc->mur', base_c, data_c)   # (M, ups_c, R)

    freq_r = fftfreq(R, uf)
    base_r = np.exp(-1j * 2 * np.pi * ar[:, None] * freq_r[None, :]).astype(cdtype)  # (ups, R)
    phase_r = np.exp(1j * 2 * np.pi * offsets[0][:, None] * freq_r[None, :]).astype(cdtype)  # (M, R)
    tmp = tmp * phase_r[:, None, :]                  # (M, ups_c, R)
    cc = np.einsum('xr,mur->mxu', base_r, tmp).conj()  # (M, ups_r, ups_c)

    flat = np.argmax(np.abs(cc).reshape(M, -1), axis=1)
    mr, mc = np.unravel_index(flat, (ups, ups))
    maxima = np.stack([mr, mc]).astype(float)        # (2, M): [rows, cols]
    maxima -= dftshift
    return shifts + maxima / uf


@dataclass
class ShiftVectors:
    r: np.ndarray
    c: np.ndarray
    shifts: np.ndarray

@dataclass
class NoShiftVectors:
    r: np.ndarray
    c: np.ndarray

def reconcile_shift_vectors(region_defn: np.ndarray, 
        xn: ShiftVectors, yn: ShiftVectors, 
        dn: ShiftVectors, an: ShiftVectors,
        bad_xn: NoShiftVectors, bad_yn: NoShiftVectors,
        shift_xn_init: np.ndarray, shift_yn_init: np.ndarray,
        no_shift_weight: float = 0.3, diagonal_weight: float = 0.5,
        learning_rates: tuple[float, int] | list[tuple[float, int]] = [(3e-1, 1000), (1e-1, 1000), (3e-2, 500)]):
    """
    Returns
    -------
    r_patch : ndarray
    c_patch : ndarray
        Vectors of the row and column indices of the N patches
    optimized_pos : ndarray
        2xN array of optimized patch coordinates where optimized_pos[0] corresponds to y-coordinates
    """
    
    npos = np.sum(region_defn)

    # two kinds of patch indices:
    #   1) 2D index (row, column) into original array of patches
    #   2) 1D index into list of patches for this region
    r_patch, c_patch = np.nonzero(region_defn)
    idx_2D_to_1D = {}
    for k, (r, c) in enumerate(zip(r_patch, c_patch)):
        idx_2D_to_1D[(r, c)] = k

    xn_indices = list(itertools.chain(zip(xn.r, xn.c), zip(bad_xn.r, bad_xn.c)))
    yn_indices = list(itertools.chain(zip(yn.r, yn.c), zip(bad_yn.r, bad_yn.c)))

    idx1_xn = np.array([idx_2D_to_1D[r  , c]   for r, c in xn_indices])
    idx2_xn = np.array([idx_2D_to_1D[r  , c+1] for r, c in xn_indices])
    idx1_yn = np.array([idx_2D_to_1D[r  , c]   for r, c in yn_indices])
    idx2_yn = np.array([idx_2D_to_1D[r+1, c]   for r, c in yn_indices])
    idx1_dn = np.array([idx_2D_to_1D[r  , c]   for r, c in zip(dn.r, dn.c)])
    idx2_dn = np.array([idx_2D_to_1D[r+1, c+1] for r, c in zip(dn.r, dn.c)])
    idx1_an = np.array([idx_2D_to_1D[r  , c+1] for r, c in zip(an.r, an.c)])
    idx2_an = np.array([idx_2D_to_1D[r+1, c]   for r, c in zip(an.r, an.c)])

    n_xn_good = len(xn.r)
    n_xn_bad = len(bad_xn.r)
    n_yn_good = len(yn.r)
    n_yn_bad = len(bad_yn.r)

    fixed_pos_weight = 1/npos

    xn_shifts = np.zeros((2, n_xn_good + n_xn_bad), dtype=float)
    xn_weights = np.ones((2, n_xn_good + n_xn_bad), dtype=float)
    xn_shifts[:, :n_xn_good] = xn.shifts
    xn_shifts[:, n_xn_good:] = shift_xn_init[:, np.newaxis]
    xn_weights[:, n_xn_good:] = no_shift_weight

    yn_shifts = np.zeros((2, n_yn_good + n_yn_bad), dtype=float)
    yn_weights = np.zeros((2, n_yn_good + n_yn_bad), dtype=float)
    yn_shifts[:, :n_yn_good] = yn.shifts
    yn_shifts[:, n_yn_good:] = shift_yn_init[:, np.newaxis]
    yn_weights[:, n_yn_good:] = no_shift_weight

    # pos[0, :] are y-coordinates
    # pos[1, :] are x-coordinates
    pos = r_patch[np.newaxis, :] * shift_yn_init[:, np.newaxis] + \
        c_patch[np.newaxis, :] * shift_xn_init[:, np.newaxis]
    avg_pos = np.mean(pos, axis=1)
    pos -= avg_pos[:, np.newaxis]

    # ------------------------------------------------------------------
    # Solve for the patch positions directly.
    #
    # The objective is a weighted sum of squared differences between the
    # optimized positions and the measured shift vectors, plus a weak anchor
    # tying the first patch to its initial position:
    #
    #   L(p) = sum_e w_e * (p[i2_e] - p[i1_e] - shift_e)^2
    #          + fixed_pos_weight * (p[0] - p_init[0])^2
    #
    # This is a convex quadratic (a weighted graph-Laplacian least-squares
    # problem) whose exact minimizer solves the normal equations  A p = b.
    # The previous implementation approximated this minimizer with ~2500
    # iterations of Adam; solving the sparse linear system directly gives the
    # exact optimum orders of magnitude faster and matches the iterated result
    # to well under a hundredth of a pixel.
    #
    # The per-edge weight is identical for the y- and x-coordinate rows, so the
    # two coordinates share a single system matrix A (one factorization, two
    # right-hand-side solves).
    # ------------------------------------------------------------------
    pos_init = pos.copy()
    target0 = pos_init[:, 0].copy()

    # Assemble every edge (x, y, diagonal, anti-diagonal) into flat arrays.
    idx1 = np.concatenate([idx1_xn, idx1_yn, idx1_dn, idx1_an])
    idx2 = np.concatenate([idx2_xn, idx2_yn, idx2_dn, idx2_an])
    # Per-edge target shift for each coordinate row -> (2, E).
    targets = np.concatenate([xn_shifts, yn_shifts, dn.shifts, an.shifts], axis=1)
    n_dn = dn.shifts.shape[1]
    n_an = an.shifts.shape[1]
    # Per-edge weight (same for both coordinate rows).
    edge_w = np.concatenate([
        xn_weights[0],
        yn_weights[0],
        np.full(n_dn, diagonal_weight, dtype=float),
        np.full(n_an, diagonal_weight, dtype=float),
    ])

    npos_i = int(npos)

    # Weighted graph Laplacian: A[i1,i1]+=w, A[i2,i2]+=w, A[i1,i2]-=w, A[i2,i1]-=w
    rows = np.concatenate([idx1, idx2, idx1, idx2])
    cols = np.concatenate([idx1, idx2, idx2, idx1])
    vals = np.concatenate([edge_w, edge_w, -edge_w, -edge_w])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(npos_i, npos_i)).tocsr()

    # Weak anchor on the first patch (matches fixed_pos_weight term).
    anchor = np.zeros(npos_i)
    anchor[0] = fixed_pos_weight
    # Tiny ridge toward the initial guess. This keeps the system positive
    # definite even if a patch (or floating sub-cluster) is left unconstrained
    # by the measured shift vectors, and reproduces Adam's behavior of leaving
    # such patches at their initial position. It is far below the anchor weight,
    # so well-constrained positions are unaffected (< 1e-6 px).
    ridge = 1e-6
    A = A + sp.diags(anchor + ridge)

    solve = spla.factorized(A.tocsc())
    optimized_pos = np.empty((2, npos_i), dtype=float)
    for d in range(2):
        wt = edge_w * targets[d]
        b = np.zeros(npos_i)
        np.add.at(b, idx2, wt)
        np.add.at(b, idx1, -wt)
        b[0] += fixed_pos_weight * target0[d]
        b += ridge * pos_init[d]
        optimized_pos[d] = solve(b)

    loss_values = []
    return r_patch, c_patch, optimized_pos, loss_values


def perfect_grid_stitching(data: np.ndarray, 
        shift_xn_guess: np.ndarray, shift_yn_guess: np.ndarray):
    nr, nc = data.shape[:2]

    r_patch = np.arange(nr).repeat(nc)
    c_patch = np.tile(np.arange(nc), nr)
    pos = r_patch[np.newaxis, :] * shift_yn_guess[:, np.newaxis] + \
            c_patch[np.newaxis, :] * shift_xn_guess[:, np.newaxis]
    stitched_region = stitch_patches(data, r_patch, c_patch, pos)
    stitched_image = stitched_region.masked_image()

    return StitchResult(stitched_image,
        stitched_region.image, stitched_region.support, 
        pos, r_patch, c_patch, 0)


@dataclass
class StitchedPatches:
    image: np.ndarray
    support: np.ndarray
    positions: np.ndarray
    r_patch: np.ndarray
    c_patch: np.ndarray

    def masked_image(self, support_threshold=0.1):
        masked_support = self.support.copy()
        masked_support[masked_support < support_threshold] = np.nan
        return self.image / masked_support
    

def _batch_shift_bilinear(imgs, shifts):
    """Batched order-1 (bilinear) image shift matching ``scipy.ndimage.shift``.

    Reproduces ``ndi.shift(img, shift, order=1, mode='constant', cval=0)`` for a
    whole stack of shifts at once, without a Python loop.

    Parameters
    ----------
    imgs : ndarray
        Either an ``(N, H, W)`` stack (one image per shift) or a single
        ``(H, W)`` image that is shifted by every shift vector.
    shifts : ndarray
        ``(N, 2)`` array of ``(dy, dx)`` shifts.

    Returns
    -------
    ndarray
        ``(N, H, W)`` array of shifted images.
    """
    imgs = np.asarray(imgs, dtype=float)
    single = imgs.ndim == 2
    H, W = imgs.shape[-2:]
    shifts = np.asarray(shifts, dtype=float)
    n = shifts.shape[0]

    sy = shifts[:, 0].reshape(n, 1, 1)
    sx = shifts[:, 1].reshape(n, 1, 1)
    oy = np.arange(H).reshape(1, H, 1)
    ox = np.arange(W).reshape(1, 1, W)
    ys = oy - sy                              # source coords (n, H, 1)
    xs = ox - sx                              # source coords (n, 1, W)

    # scipy 'constant' mode returns cval (0) wherever the *source* coordinate
    # leaves [0, size-1]; inside that range it is ordinary bilinear interpolation.
    valid = (ys >= 0) & (ys <= H - 1) & (xs >= 0) & (xs <= W - 1)

    ysc = np.clip(ys, 0, H - 1)
    xsc = np.clip(xs, 0, W - 1)
    y0 = np.floor(ysc).astype(np.int64)
    x0 = np.floor(xsc).astype(np.int64)
    fy = ysc - y0
    fx = xsc - x0
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)

    y0b, y1b = np.broadcast_to(y0, (n, H, W)), np.broadcast_to(y1, (n, H, W))
    x0b, x1b = np.broadcast_to(x0, (n, H, W)), np.broadcast_to(x1, (n, H, W))
    fyb, fxb = np.broadcast_to(fy, (n, H, W)), np.broadcast_to(fx, (n, H, W))

    if single:
        def gather(yy, xx):
            return imgs[yy, xx]
    else:
        nidx = np.arange(n).reshape(n, 1, 1)

        def gather(yy, xx):
            return imgs[nidx, yy, xx]

    out = ((1 - fyb) * (1 - fxb) * gather(y0b, x0b)
           + (1 - fyb) * fxb * gather(y0b, x1b)
           + fyb * (1 - fxb) * gather(y1b, x0b)
           + fyb * fxb * gather(y1b, x1b))
    out[~valid] = 0.0
    return out


def stitch_patches(data, r_patch, c_patch, patch_pos):
    """
    Parameters
    ----------
    data : ndarray
        nr x nc x p x p array of image patches
    r_patch : ndarray
    c_patch : ndarray
        Vectors of the row and column indices of the N patches
    patch_pos : ndarray
        2xN array of patch coordinates where optimized_pos[0] corresponds to y-coordinates
    """
    padded_data = np.pad(data, ((0,0), (0,0), (1,1), (1,1)))
    im_shape = data.shape[2:]
    padded_support = np.pad(np.ones(im_shape, dtype=float), 1)
    padded_shape = np.array(padded_support.shape)

    window = signal.windows.hamming(padded_data.shape[2])
    window2 = window[:, np.newaxis] * window[np.newaxis, :]
    windowed_padded_support = padded_support * window2
    windowed_padded_data = padded_data * window2[np.newaxis, np.newaxis, :, :]

    patch_pos -= np.min(patch_pos, axis=1)[:, np.newaxis]
    region_max = np.ceil(np.max(patch_pos, axis=1)).astype(int) + padded_shape

    # positions of the top left corners of the patches
    # +1 to account for padding
    padded_pos = patch_pos + 1

    npos = len(r_patch)
    ps0, ps1 = int(padded_shape[0]), int(padded_shape[1])

    rounded_pos = np.round(patch_pos)                 # (2, npos)
    subpx_shift = (patch_pos - rounded_pos).T         # (npos, 2)

    # Shift every patch (and the shared support window) to subpixel precision
    # in one vectorized bilinear pass instead of a per-patch scipy loop.
    gathered = windowed_padded_data[r_patch, c_patch]                 # (npos, ps0, ps1)
    shifted_patches = _batch_shift_bilinear(gathered, subpx_shift)
    shifted_support = _batch_shift_bilinear(windowed_padded_support, subpx_shift)

    # Accumulate all patches onto the canvas at their (integer) positions with a
    # single scatter-add per output. Overlapping contributions sum, exactly as
    # the original slice-add loop did.
    r0 = rounded_pos[0].astype(np.int64)
    c0 = rounded_pos[1].astype(np.int64)
    ay = np.arange(ps0)
    ax = np.arange(ps1)
    iy = (r0[:, None, None] + ay[None, :, None])
    ix = (c0[:, None, None] + ax[None, None, :])
    ncols = int(region_max[1])
    flat = np.broadcast_to(iy, (npos, ps0, ps1)) * ncols \
        + np.broadcast_to(ix, (npos, ps0, ps1))
    flat = flat.reshape(-1)
    size = int(region_max[0]) * ncols
    canvas = np.bincount(flat, weights=shifted_patches.reshape(-1),
                         minlength=size).reshape(region_max)
    canvas_support = np.bincount(flat, weights=shifted_support.reshape(-1),
                                 minlength=size).reshape(region_max)

    return StitchedPatches(canvas, canvas_support, padded_pos, r_patch, c_patch)