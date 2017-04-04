import os.path
import time
from cpython cimport PyErr_CheckSignals
import numpy as np
cimport numpy as np
import cython
from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    cdef unsigned int i

    for i in range(num_infos):
        pass_filter[i] = select[0] <= infos[i].timestamp/1e9 and infos[i].timestamp/1e9 < select[1]
    return


def run(t0, t1, a=3, w=1, c=64, k=1, i=1000000, timeout=30, cfile=None):
    """ Read data between t0 and t1 with given specification.
    Parameters are like that of vyssim:
    a is number of antennas,
    w is number of spectral windows,
    c is number of channels per window,
    k is number of stokes parameters, and
    i is integration time in microseconds.
    Will return numpy array when all data collected or as much as is ready when timeout elapses.
    cfile is the vys/vysmaw configuration file.
    """

    # define time window
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array([t0, t1], dtype=np.float64)

    # configure
    cdef Configuration config
    if cfile:
        assert os.path.exists(cfile), 'Configuration file {0} not found.'.format(cfile)
        print('Reading {0}'.format(cfile))
        config = cy_vysmaw.Configuration(cfile)
    else:
        print('Not using a vys configuration file')
        config = cy_vysmaw.Configuration()

    # set windows
    cdef void **u = <void **>malloc(sizeof(void *))
    u[0] = &windows[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

    # set filters
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

    f[0] = filter_time
    handle, consumers = config.start(1, f, u)
    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue0 = c0.queue()
    cdef vysmaw_message *msg = NULL

    ni = (t1-t0)/int(i/1e6)  # t1, t0 in seconds, i in microsec
    nbl = a*(a-1)/2
    blarr = np.array([(ind0, ind1) for ind1 in range(a) for ind0 in range(0,ind1)])
    nch = w*c
    ntot = ni*nbl*w*k
    data = np.zeros(shape=(ni, nbl, nch, k), dtype='complex128')
    time0 = int(time.time())

    cdef long spec = 0
    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < ntot) and (int(time.time()) - time0 < timeout):
        if msg is not NULL:
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)

                iind = 0
#                print(np.array(py_msg.info.stations))
                bind = np.where(blarr == np.array(py_msg.info.stations))
                ch0 = c*py_msg.info.spectral_window_index # or baseband_index? or baseband_id?
                pind = py_msg.info.polarization_product_id
#                print(bind, ch0, pind)
                data[iind, bind, ch0:ch0+c, pind].real = np.array(py_msg.buffer)[::2]
                data[iind, bind, ch0:ch0+c, pind].imag = np.array(py_msg.buffer)[1::2]
                spec = spec + 1
                py_msg.unref()
            else:
#                print(str('msg: {0}'.format(msg[0].typ)))
                vysmaw_message_unref(msg)
        else:
            print('msg: NULL')
        msg = vysmaw_message_queue_timeout_pop(queue0, 1000000)
        PyErr_CheckSignals()

    if handle is not None:
        handle.shutdown()

    return data