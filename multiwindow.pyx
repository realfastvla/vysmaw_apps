from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw
import signal
import numpy as np
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    for i in range(num_infos):
        pass_filter[i] = select[0] <= infos[i].timestamp and infos[i].timestamp < select[1]
    return


cdef void filter_pol(const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    for i in range(num_infos):
        pass_filter[i] = pol == select[0]
    return


def run(*args, n_stop=10, filter='pol'):
    """ Playing...
    (assume one for now)
    """

    # define windows
    print(args, type(args))
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array(args, dtype=np.float64)
    N = len(windows)

    # configure
    cdef Configuration config
    config = cy_vysmaw.Configuration()

    # set windows
    cdef void **u = <void **>malloc(N * sizeof(void *))
    u[0] = &windows[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

    # set filters
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(N * sizeof(vysmaw_spectrum_filter))

    if filter == 'pol':
        f[0] = filter_pol
    elif filter == 'time':
        f[0] = filter_time

    handle, consumers = config.start(1, f, u)

    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue0 = c0.queue()
    cdef vysmaw_message *msg = NULL

    data = np.zeros(n_stop)
    i = 0

    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (i < n_stop):
        if msg is not NULL:
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)
#                print(str(py_msg))
                data[i] = py_msg.info.polarization_product_id
                py_msg.unref()
                i = i + 1
            else:
                vysmaw_message_unref(msg)

        msg = vysmaw_message_queue_timeout_pop(queue0, 500000)

    if handle is not None:
        handle.shutdown()

    return data