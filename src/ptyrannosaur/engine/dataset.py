"""Load 4D-STEM data for training."""

import h5py
import numpy as np
import jax.numpy as jnp
import ptyrannosaur.engine.utils as utils
from tqdm.auto import tqdm

def batch_exp(norm_dps, neighbors, scan_pts):
    """Create batch of data for experiment."""
    center_id = neighbors.shape[1]//2
    input_dps = jnp.moveaxis(norm_dps[neighbors], 1, -1)
    return input_dps, scan_pts[neighbors[:,center_id]]

def loop_batch_exp(dps, neighbors, all_scan_pts, batch_size, model, model_state,
                   on_device_gather=True):
    """Run the network over experimental data in batches.

    Parameters
    ----------
    dps : ndarray
        (num_patterns, n_k, n_k) normalized diffraction patterns.
    neighbors : ndarray
        (num_evals, num_neighbors) indices into `dps` for each output patch.
    all_scan_pts : ndarray
        Scan positions indexed by pattern.
    batch_size : int
        Number of output patches per batch. Any value works (the final batch may
        be smaller); larger batches reduce per-call overhead and improve GPU
        utilization, bounded by device memory.
    on_device_gather : bool, optional
        If True (default), move the diffraction patterns to the device once and
        gather each batch's neighbor stack there, avoiding a per-batch host build
        and host->device transfer of the ~25x-redundant input. Set False to use
        the original host-side gather path. Both produce identical outputs.

    Returns
    -------
    outputs : jax.Array
        Concatenated network outputs, one patch per eval, in `neighbors` order.
    scan_pts : jax.Array
        Center scan position for each eval.
    """
    num_evals = neighbors.shape[0]
    center_id = neighbors.shape[1] // 2
    num_batches = (num_evals + batch_size - 1) // batch_size

    outputs = []
    if on_device_gather:
        # Transfer the (unique) diffraction patterns and index table once.
        dps_dev = jnp.asarray(dps)
        neighbors_dev = jnp.asarray(neighbors)
        for n in tqdm(range(num_batches), desc="Processing batches"):
            batch_neighbors = neighbors_dev[n*batch_size:(n+1)*batch_size]
            batch_outputs = utils.eval_exp_batch_gather(
                model, model_state, dps_dev, batch_neighbors)
            outputs.append(batch_outputs)
    else:
        for n in tqdm(range(num_batches), desc="Processing batches"):
            batch_neighbors = neighbors[n*batch_size:(n+1)*batch_size]
            batch_dps, _ = batch_exp(dps, batch_neighbors, all_scan_pts)
            batch_outputs = utils.eval_exp_batch(model, model_state, batch_dps)
            outputs.append(batch_outputs)

    # Center scan positions in eval order (matches the concatenated outputs).
    scan_pts = jnp.asarray(all_scan_pts)[jnp.asarray(neighbors[:, center_id])]
    return jnp.concatenate(outputs), scan_pts