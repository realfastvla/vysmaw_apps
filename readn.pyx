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


message_types = dict(zip([VYSMAW_MESSAGE_VALID_BUFFER, VYSMAW_MESSAGE_ID_FAILURE, VYSMAW_MESSAGE_QUEUE_ALERT, 
          VYSMAW_MESSAGE_DATA_BUFFER_STARVATION, VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION, 
          VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE, VYSMAW_MESSAGE_RDMA_READ_FAILURE, VYSMAW_MESSAGE_VERSION_MISMATCH,
          VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW, VYSMAW_MESSAGE_END],
          ["VYSMAW_MESSAGE_VALID_BUFFER", "VYSMAW_MESSAGE_ID_FAILURE", "VYSMAW_MESSAGE_QUEUE_ALERT", 
          "VYSMAW_MESSAGE_DATA_BUFFER_STARVATION", "VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION", 
          "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE", "VYSMAW_MESSAGE_RDMA_READ_FAILURE", "VYSMAW_MESSAGE_VERSION_MISMATCH",
          "VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW", "VYSMAW_MESSAGE_END"]))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef unsigned int i

    for i in range(num_infos):
        pass_filter[i] = True

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(n_stop, cfile=None):

    # configure
    cdef Configuration config
    if cfile:
        print('Reading {0}'.format(cfile))
        config = cy_vysmaw.Configuration(cfile)
    else:
        print('Using default vys configuration file')
        config = cy_vysmaw.Configuration()

    specbytes = vys_max_spectrum_buffer_size(1024, 1)
    config.max_spectrum_buffer_size = specbytes
    config.spectrum_buffer_pool_size = n_stop*specbytes

    cdef unsigned int num_spectra = 0
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

    f[0] = filter
    handle, consumers = config.start(1, f, NULL)

    free(f)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue = c0.queue()
    cdef vysmaw_message *msg = NULL
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] typecount = np.zeros(10, dtype=np.int)

    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (num_spectra < n_stop):
        msg = vysmaw_message_queue_timeout_pop(queue, 100000)

        if msg is not NULL:
            num_spectra += 1
            typecount[msg[0].typ] += 1
            vysmaw_message_unref(msg)

        PyErr_CheckSignals()

    if handle is not None:
        handle.shutdown()

    return dict(zip(message_types.values(), typecount))
