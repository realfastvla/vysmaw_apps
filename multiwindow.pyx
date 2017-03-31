from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw
import signal
import numpy as np


cdef void cb(const uint8_t *stns, uint8_t spw, uint8_t bb, uint8_t pol,
             const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef float *limits = <float *>user_data
    for i in range(pass_filter.shape[0]):
        pass_filter[i] = limits[0] <= infos[i].timestamp and infos[i].timestamp < limits[1]
    return


def run(start, end, rate=1):
    cdef Configuration config
    config = cy_vysmaw.Configuration()

    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

    N = 1  # number of consumers
    cdef void **u = <void **>malloc(N * sizeof(void *))

    f[0] = cb
    u[0] = np.array([start, stop])
    handle, consumers = config.start(1, f, u)

    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue = c0.queue()
    cdef vysmaw_message *msg = NULL

    n_stop = (end-start)/rate
    times = np.zeros(n_stop)
    i = 0

    while (msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END):
        if msg is not NULL:
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)
                print(str(py_msg))
                times[i] = py_msg.info.timestamp/1e9
                py_msg.unref()
                i = i + 1
            else:
                vysmaw_message_unref(msg)

        msg = vysmaw_message_queue_timeout_pop(queue, 500000)

    if handle is not None:
        handle.shutdown()

    return times