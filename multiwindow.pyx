from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw
import signal
import numpy as np

def filter_timewindow(start, end):
#    cdef void cb(const uint8_t *stns, uint8_t spw, uint8_t sto, const vys_spectrum_info *infos, uint8_t num_infos, void *user_data, bool *pass_filter) nogil:
    def cb(stns, spw, sto, infos, num_infos, user_data, pass_filter):
        for i in range(pass_filter.shape[0]):
            pass_filter[i] = start <= infos[i].timestamp and infos[i].timestamp < end
        return
    return cb

cdef void cb(const uint8_t *stns, uint8_t spw, uint8_t sto, const vys_spectrum_info *infos, uint8_t num_infos, void *user_data, bool *pass_filter) nogil:
    for i in range(pass_filter.shape[0]):
        pass_filter[i] = 0 <= infos[i].timestamp and infos[i].timestamp < 1
    return


def run(start, end, rate=1):
    cdef Configuration config
    config = cy_vysmaw.Configuration()

    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))
#    cb = filter_timewindow(start, end)
    f[0] = cb

    handle, consumers = config.start(1, f, NULL)

    free(f)

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