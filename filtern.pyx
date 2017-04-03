from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw
import signal
import numpy as np
cimport numpy as np
import cython
from cpython cimport PyErr_CheckSignals
import os.path

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    for i in range(num_infos):
        pass_filter[i] = select[0] <= infos[i].timestamp/1e9 and infos[i].timestamp/1e9 < select[1]
    return


cdef void filter_pol(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    for i in range(num_infos):
        pass_filter[i] = pol == select[0]
    return


cdef void filter_none(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    for i in range(num_infos):
        pass_filter[i] = True
    return


def run(*args, filter='', nr=1, nch=64, cfile=None):
    """ Read nr filtered messages and return values appropriate for given filter.
    filters can be empty, 'pol', 'time'.
    pol needs a integer pol value.
    time needs two unix times in seconds.
    empty needs no argument.
    nch in number of channels.
    cfile is the vys/vysmaw configuration file
    """

    # define windows
#    print(args, type(args))
    if not len(args): # catch if no args passed for filter=''
        args = [0]
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array(args, dtype=np.float64)
    N = len(windows)

    # configure
    cdef Configuration config
    if cfile:
        assert os.path.exists(cfile), 'Configuration file {0} not found.'.format(cfile)
        config = cy_vysmaw.Configuration(cfile)
    else:
        config = cy_vysmaw.Configuration()

    # set windows
    cdef void **u = <void **>malloc(N * sizeof(void *))
    u[0] = &windows[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

    # set filters
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(N * sizeof(vysmaw_spectrum_filter))

    if not filter:
        f[0] = filter_none
    elif filter == 'pol':
        f[0] = filter_pol
    elif filter == 'time':
        f[0] = filter_time
    else:
        print('filter name not recognized')
        free(f)
        free(u)
        return

    handle, consumers = config.start(1, f, u)

    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue0 = c0.queue()
    cdef vysmaw_message *msg = NULL

    data = np.zeros(shape=(nr, nch), dtype='complex128')

    cdef long i = 0
    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (i < nr):
        if msg is not NULL:
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)
                data[i].real = np.array(py_msg.buffer)[::2]
                data[i].imag = np.array(py_msg.buffer)[1::2]
                i = i + 1
                py_msg.unref()
            else:
                vysmaw_message_unref(msg)

        msg = vysmaw_message_queue_timeout_pop(queue0, 500000)
        PyErr_CheckSignals()

    if handle is not None:
        handle.shutdown()

    return data