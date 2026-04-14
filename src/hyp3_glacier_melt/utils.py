import numpy as np
import re

# faster alternative to np.nanpercentile -- from https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
def nan_percentile(arr, q, axis=0):
    arr = np.asarray(arr)
    # Move the desired axis to the front
    arr = np.moveaxis(arr, axis, 0)

    # valid (non-NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    max_val = np.nanmax(arr)
    arr = arr.copy()  # avoid modifying original input
    arr[np.isnan(arr)] = max_val
    arr = np.sort(arr, axis=0)

    # Handle list or single q
    qs = [q] if np.isscalar(q) else q
    result = []

    for quant in qs:
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        floor_val = _zvalue_from_index(arr, f_arr) * (c_arr - k_arr)
        ceil_val  = _zvalue_from_index(arr, c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr, k_arr.astype(np.int32))[fc_equal_k_mask]

        result.append(quant_arr)

    if np.isscalar(q):
        return result[0]
    return np.stack(result, axis=0)

def _zvalue_from_index(arr, ind):
    """
    Extracts values along the first axis of a 3D array given 2D indices.
    """
    # arr shape = (depth, y, x)
    d, nC, nR = arr.shape
    # Compute linear indices
    idx = nC*nR*ind + nR*np.arange(nC)[:, None] + np.arange(nR)
    return np.take(arr, idx)
    
class DescStr:
    def __init__(self):
        self._desc = ''
    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)
    def read(self):
        ret = self._desc
        self._desc = ''
        return ret
    def flush(self):
        pass
