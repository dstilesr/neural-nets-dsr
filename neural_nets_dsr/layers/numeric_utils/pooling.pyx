cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange

ctypedef np.ndarray ARR
ctypedef np.float64_t NPFLOAT
ctypedef NPFLOAT (*agg_2d) (NPFLOAT[:, :]) nogil


@cython.wraparound(False)
cdef void multiply_2d_inplace(NPFLOAT[:, :] aslice, NPFLOAT num) nogil:
    """
    Multiply a 2d slice by a number in place.
    :param aslice: A 2D slice.
    :param num: Number by which to multiply.
    :param out: Slice to store the output. Must have same shape as aslice.
    :return: 
    """
    cdef int ilim = aslice.shape[0], jlim = aslice.shape[1]
    cdef int i, j

    for i in range(ilim):
        for j in range(jlim):
            aslice[i, j] *= num

@cython.wraparound(False)
cdef NPFLOAT slice_max(NPFLOAT[:, :] slc) nogil:
    """
    Computes the maximum of the given 2D slice.
    :param slc: 
    :return: 
    """
    cdef int i = 0, j = 0, ilim = slc.shape[0], jlim = slc.shape[1]
    cdef NPFLOAT maximum = slc[i, j]
    cdef NPFLOAT temp

    for i in range(ilim):
        for j in range(jlim):
            temp = slc[i, j]
            if temp > maximum:
                maximum = temp

    return maximum


@cython.wraparound(False)
cdef NPFLOAT slice_avg(NPFLOAT[:, :] slc) nogil:
    """
    Computes the maximum of the given 2D slice.
    :param slc: 
    :return: 
    """
    cdef int i = 0, j = 0, ilim = slc.shape[0], jlim = slc.shape[1]
    cdef NPFLOAT slc_sum = 0.0
    cdef NPFLOAT num_entries = ilim * jlim

    for i in range(ilim):
        for j in range(jlim):
            slc_sum += slc[i, j]

    return slc_sum / num_entries


cdef void pool2d_slice(
        NPFLOAT[:, :] x_slice,
        int sizex,
        int sizey,
        agg_2d agg_func,
        NPFLOAT[:, :] out) nogil:
    """
    Auxiliary function to perform pooling on a 2D slice.
    :param x_slice: 
    :param sizex: Filter height.
    :param sizey: Filter width.
    :param agg_func: Function to be used for aggregation.
    :param out: Slice to store output.
    :return: None
    """
    cdef int xlim = out.shape[0], ylim = out.shape[1]
    cdef int i, j, ilo, ihi, jlo, jhi
    for i in range(xlim):
        ilo = i * sizex
        ihi = ilo + sizex
        for j in range(ylim):
            jlo = j * sizey
            jhi = jlo + sizey
            out[i, j] = agg_func(x_slice[ilo:ihi, jlo:jhi])


@cython.wraparound(False)
cdef ARR[NPFLOAT, ndim=4] pool_2d(
        ARR[NPFLOAT, ndim=4] x,
        int sizex,
        int sizey,
        agg_2d agg_func):
    """
    Perform 2D max pooling on an array. Stride is taken to be the size of the
    filter.
    :param x: Input volume.
    :param sizex: x size for max pooling filter.
    :param sizey: y size for max pooling filter.
    :param agg_func: Function used to aggregate over 2D slices.
    :return: Reduced volume.
    """
    cdef int elim = x.shape[0], channels = x.shape[3]
    cdef int xlim = x.shape[1] // sizex, ylim = x.shape[2] // sizey
    cdef ARR[NPFLOAT, ndim=4] output = np.zeros((elim, xlim, ylim, channels))

    cdef NPFLOAT[:, :, :, :] xview = x, outview = output
    cdef int e, k
    for e in prange(elim, nogil=True):
        for k in range(channels):
            pool2d_slice(
                xview[e, :, :, k],
                sizex,
                sizey,
                agg_func,
                outview[e, :, :, k]
            )

    return output


@cython.wraparound(False)
cpdef ARR[NPFLOAT, ndim=4] max_pool_2d(
        ARR[NPFLOAT, ndim=4] x,
        int sizex,
        int sizey):
    """
    Perform 2D max pooling on an array. Stride is taken to be the size of the
    filter.
    :param x: Input volume.
    :param sizex: x size for max pooling filter.
    :param sizey: y size for max pooling filter.
    :return: Reduced volume.
    """
    return pool_2d(x, sizex, sizey, slice_max)


@cython.wraparound(False)
cpdef ARR[NPFLOAT, ndim=4] avg_pool_2d(
        ARR[NPFLOAT, ndim=4] x,
        int sizex,
        int sizey):
    """
    Perform 2D max pooling on an array. Stride is taken to be the size of the
    filter.
    :param x: Input volume.
    :param sizex: x size for max pooling filter.
    :param sizey: y size for max pooling filter.
    :return: Reduced volume.
    """
    return pool_2d(x, sizex, sizey, slice_avg)


@cython.wraparound(False)
cpdef ARR[NPFLOAT, ndim=4] expand_pooled(
        ARR[NPFLOAT, ndim=4] x,
        int out_x,
        int out_y,
        int filtx,
        int filty):
    """
    Expands a pooled layer for backprop.
    :param x: Array.
    :param out_x: Height of output array.
    :param out_y: Width of output array.
    :param filtx: Height of pool filter.
    :param filty: Width of pool filter.
    :return: 
    """
    cdef int elim = x.shape[0], channels = x.shape[3]
    cdef ARR[NPFLOAT, ndim=4] out = np.ones((elim, out_x, out_y, channels))
    cdef int xlim = out_x // filtx, ylim = out_y // filty
    cdef NPFLOAT[:, :, :, :] outview = out, xview = x

    cdef int i, j, e, k, ilo, ihi, jlo, jhi
    for e in prange(elim, nogil=True):
        for k in prange(channels):
            for i in range(xlim):
                ilo = i * filtx
                ihi = ilo + filtx
                for j in range(ylim):
                    jlo = j * filty
                    jhi = jlo + filty
                    multiply_2d_inplace(
                        outview[e, ilo:ihi, jlo:jhi, k],
                        x[e, i, j, k]
                    )

    out[:, (xlim * filtx + 1):, :, :] = 0.0
    out[:, :, (ylim * filty + 1):, :] = 0.0
    return out
