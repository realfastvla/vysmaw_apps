# cython: language_level=2
import cython
from libc.stdint cimport *
from libc.stdlib cimport *

from cpython cimport PyErr_CheckSignals
import signal
import numpy as np
cimport numpy as np

from vysmaw import cy_vysmaw
from vysmaw.cy_vysmaw cimport *
from vysmaw.vysmaw cimport *

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

    cdef unsigned int num_spectra = 0
    handle, consumers = config.start(filter, NULL)

    cdef Consumer c0 = consumers
    cdef vysmaw_message_queue queue = c0.queue()
    cdef vysmaw_message *msg = NULL
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] typecount = np.zeros(7, dtype=np.int)

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