cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange

ctypedef np.ndarray ARR
ctypedef np.float64_t NPFLOAT


@cython.wraparound(False)
cdef NPFLOAT slice_prod(NPFLOAT[:, :] a, NPFLOAT[:, :] b) nogil:
    """
    Return the sum of the element-wise product of two 2D slices.
    :param a: 
    :param b: 
    :return: 
    """
    cdef NPFLOAT out = 0.0
    cdef int i, j, a0, a1
    a0 = a.shape[0]
    a1 = a.shape[1]

    for i in range(a0):
        for j in range(a1):
            out += a[i, j] * b[i, j]

    return out


@cython.wraparound(False)
cdef NPFLOAT[:, :] filter_conv(NPFLOAT[:, :, :] vol, NPFLOAT[:, :, :] filt, NPFLOAT[:, :] out) nogil:
    """
    Convolve a filter with an input volume.
    :param vol: Input volume.
    :param filt: Filter.
    :param out: 2D array view to store output.
    :return:
    """
    cdef int v0 = vol.shape[0], v1 = vol.shape[1], in_channels = vol.shape[2]
    cdef int f0 = filt.shape[0], f1 = filt.shape[1]
    cdef int xlim = v0 - 2 * (f0 // 2), ylim = v1 - 2 * (f1 // 2)

    cdef NPFLOAT temp
    cdef int i, j, iup, jup, k
    for i in range(xlim):
        iup = i + f0
        for j in range(ylim):
            jup = j + f1
            temp = 0.0
            for k in range(in_channels):
                temp += slice_prod(vol[i:iup, j:jup, k], filt[:, :, k])

            out[i, j] = temp
    return out


@cython.wraparound(False)
def full_conv(ARR[NPFLOAT, ndim=4] vol, ARR[NPFLOAT, ndim=4] filt):
    """
    Compute the full convolution of the input volume and the filter set for
    convolutional layers.
    :param vol: Volume to process.
    :param filt: Filter array. Shape: (filt_height, filt_width,
        previous_channels, num_filters)
    :return:
    """
    cdef int num_examples = vol.shape[0]
    cdef int in_chan = vol.shape[3]
    cdef int out_chan = filt.shape[3]
    cdef int f0 = filt.shape[0], f1 = filt.shape[1]
    cdef int xshape = vol.shape[1] - 2 * (f0 // 2)
    cdef int yshape = vol.shape[2] - 2 * (f1 // 2)

    cdef int i, j, itemp, jtemp, e, l
    cdef NPFLOAT temp
    cdef ARR[NPFLOAT, ndim=4] output = np.zeros(
        [num_examples, xshape, yshape, out_chan],
        dtype=np.float64
    )
    cdef NPFLOAT[:, :, :, :] out_view = output, vview = vol, fview = filt

    # Loop over training examples
    for e in prange(num_examples, nogil=True):
        # Loop over filters
        for l in prange(out_chan):
            out_view[e, :, :, l] = filter_conv(
                vview[e, :, :, :],
                fview[:, :, :, l],
                out_view[e, :, :, l]
            )
    return output
