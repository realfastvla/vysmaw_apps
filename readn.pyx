from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
from cpython cimport PyErr_CheckSignals
import cy_vysmaw
import signal
import numpy as np
cimport numpy as np
import cython


message_types = dict(zip([VYSMAW_MESSAGE_VALID_BUFFER, VYSMAW_MESSAGE_ID_FAILURE,
	      VYSMAW_MESSAGE_QUEUE_OVERFLOW, VYSMAW_MESSAGE_DATA_BUFFER_STARVATION,
	      VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION, VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE,
	      VYSMAW_MESSAGE_RDMA_READ_FAILURE, VYSMAW_MESSAGE_VERSION_MISMATCH, 
	      VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW, VYSMAW_MESSAGE_END],
	      ["VYSMAW_MESSAGE_VALID_BUFFER", "VYSMAW_MESSAGE_ID_FAILURE",
	      "VYSMAW_MESSAGE_QUEUE_OVERFLOW", "VYSMAW_MESSAGE_DATA_BUFFER_STARVATION",
	      "VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION", "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE",
	      "VYSMAW_MESSAGE_RDMA_READ_FAILURE", "VYSMAW_MESSAGE_VERSION_MISMATCH",
	      "VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW", "VYSMAW_MESSAGE_END"]))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    cdef unsigned int i

    for i in range(num_infos):
        ts = infos[i].timestamp/1e9
        if select[0] <= ts and ts < select[1]:
            pass_filter[i] = True

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(n_stop, t0, t1, cfile=None):

    # configure
    cdef Configuration config
    if cfile:
        print('Reading {0}'.format(cfile))
        config = cy_vysmaw.Configuration(cfile)
    else:
        print('Using default vys configuration file')
        config = cy_vysmaw.Configuration()

    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array([t0, t1], dtype=np.float64)
    cdef void **u = <void **>malloc(sizeof(void *))
    u[0] = &windows[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

    cdef unsigned int num_spectra = 0
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))
 
#    f[0] = cb
    f[0] = filter_time
#    handle, consumers = config.start(1, f, NULL)
    handle, consumers = config.start(1, f, u)

    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue = c0.queue()
    cdef vysmaw_message *msg = NULL
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] typecount = np.zeros(9, dtype=np.int)

    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (num_spectra < n_stop):
        msg = vysmaw_message_queue_timeout_pop(queue, 100000)

        if msg is not NULL:
            num_spectra += 1
            typecount[msg[0].typ] += 1
            vysmaw_message_unref(msg)
            
        PyErr_CheckSignals()

    if handle is not None:
        handle.shutdown()

    return typecount