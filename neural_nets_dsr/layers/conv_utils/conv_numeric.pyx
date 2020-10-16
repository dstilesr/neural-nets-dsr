import numpy as np
cimport numpy as np


ctypedef np.ndarray ARR
ctypedef np.float64_t NPFLOAT


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

    cdef int i, j, itemp, jtemp, i2, j2, e, k, l
    cdef NPFLOAT temp
    cdef ARR[NPFLOAT, ndim=4] output = np.zeros(
        [num_examples, xshape, yshape, out_chan],
        dtype=np.float64
    )

    # Loop over training examples
    for e in range(num_examples):
        # Loop over filters
        for l in range(out_chan):
            # Loop over output volume
            for i in range(xshape):
                for j in range(yshape):
                    temp = 0.0
                    # Filter convolution
                    for i2 in range(f0):
                        itemp = i + i2
                        for j2 in range(f1):
                            jtemp = j + j2
                            for k in range(in_chan):
                                temp += vol[e, itemp, jtemp, k] * filt[i2, j2, k, l]

                    output[e, i, j, l] = temp
    return output
