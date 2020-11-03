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
    :param a: A 2D slice.
    :param b: Another 2D slice. Must have the same shape as a.
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
cpdef ARR[NPFLOAT, ndim=4] full_conv(ARR[NPFLOAT, ndim=4] vol, ARR[NPFLOAT, ndim=4] filt):
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


@cython.wraparound(False)
cdef NPFLOAT[:, :, :] multiply_3d(
        NPFLOAT[:, :, :] aslice,
        NPFLOAT num,
        NPFLOAT[:, :, :] out) nogil:
    """
    Multuiply a 3d slice by a number.
    :param aslice: A 3D slice.
    :param num: Number by which to multiply.
    :param out: Slice to store the output. Must have same shape as aslice.
    :return: 
    """
    cdef int ilim = aslice.shape[0], jlim = aslice.shape[1], klim = aslice.shape[2]
    cdef int i, j, k

    for i in range(ilim):
        for j in range(jlim):
            for k in range(klim):
                out[i, j, k] = aslice[i, j, k] * num

    return out


@cython.wraparound(False)
cdef void add_to_slice_3(
        NPFLOAT[:, :, :] s1,
        NPFLOAT[:, :, :] s2) nogil:
    """
    Add the 3d slice s2 to the 3d slice s1 inplace.
    :param s1: 
    :param s2:
    :return: 
    """
    cdef int ilim = s1.shape[0], jlim = s1.shape[1], klim = s1.shape[2]
    cdef int i, j, k

    for i in range(ilim):
        for j in range(jlim):
            for k in range(klim):
                s1[i, j, k] += s2[i, j, k]


@cython.wraparound(False)
def conv_backprop(
        ARR[NPFLOAT, ndim=4] dz,
        ARR[NPFLOAT, ndim=4] filters,
        ARR[NPFLOAT, ndim=4] aprev):
    """
    Compute derivatives of gradient with respect to the filters of a
    convolutional layer and the previous layer's activations.
    :param dz: Gradient wrt
    :param filters: Filters of this layer.
    :param aprev: Activations of the previous layer.
    :return: The gradients dw, daprev.
    """
    cdef ARR[NPFLOAT, ndim=4] daprev = np.zeros_like(aprev)
    cdef ARR[NPFLOAT, ndim=4] dw = np.zeros_like(filters)

    cdef NPFLOAT[:, :, :, :] dzview = dz, dapview = daprev, dfview = dw
    cdef NPFLOAT[:, :, :, :] fview = filters, apview = aprev

    # Shapes to limit the various loops
    cdef int f0 = filters.shape[0], f1 = filters.shape[1]
    cdef int elim = dz.shape[0]
    cdef int ilim = aprev.shape[1] - 2 * (f0 // 2)
    cdef int jlim = aprev.shape[2] - 2 * (f1 // 2)
    cdef int klim = filters.shape[3], prev_chan = filters.shape[2]

    # looping and temp vars
    cdef int e, i, j, k, iup, jup
    cdef NPFLOAT temp

    # To temporarily store results of arithmetic
    cdef NPFLOAT[:, :, :, :] tempslicedf = np.zeros((elim, f0, f1, prev_chan))

    for e in prange(elim, nogil=True):
        for k in range(klim):
            for i in range(ilim):
                iup = i + f0
                for j in range(jlim):
                    jup = j + f1
                    temp = dzview[e, i, j, k]
                    add_to_slice_3(
                        dfview[:, :, :, k],
                        multiply_3d(
                            apview[e, i:iup, j:jup, :],
                            temp,
                            tempslicedf[e, :, :, :]
                        )
                    )
                    add_to_slice_3(
                        dapview[e, i:iup, j:jup, :],
                        multiply_3d(
                            fview[:, :, :, k],
                            temp,
                            tempslicedf[e, :, :, :]
                        )
                    )

    return dw, daprev




