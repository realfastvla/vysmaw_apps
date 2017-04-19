from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw
import signal
import numpy as np

cdef void cb(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol,
             const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef float *ncb = <float *>user_data
    for i in range(num_infos):
        pass_filter[i] = True
    ncb[0] += 1
    return


message_types = dict(zip([VYSMAW_MESSAGE_VALID_BUFFER, VYSMAW_MESSAGE_DIGEST_FAILURE,
    VYSMAW_MESSAGE_QUEUE_OVERFLOW, VYSMAW_MESSAGE_DATA_BUFFER_STARVATION,
    VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION, VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE,
    VYSMAW_MESSAGE_RDMA_READ_FAILURE, VYSMAW_MESSAGE_END],
    ["VYSMAW_MESSAGE_VALID_BUFFER", "VYSMAW_MESSAGE_DIGEST_FAILURE",
    "VYSMAW_MESSAGE_QUEUE_OVERFLOW", "VYSMAW_MESSAGE_DATA_BUFFER_STARVATION",
    "VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION", "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE",
    "VYSMAW_MESSAGE_RDMA_READ_FAILURE", "VYSMAW_MESSAGE_END"]))


def run(n_stop):
    cdef Configuration config
    config = cy_vysmaw.Configuration()
    cdef unsigned long num_cbs = 0

    num_spectra = 0

    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))
    f[0] = cb
    cdef void **u = <void **>malloc(sizeof(void *))
    u[0] = &num_cbs

    handle, consumers = config.start(1, f, u)

    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef vysmaw_message_queue queue = c0.queue()
    cdef vysmaw_message *msg = NULL

    bls = np.zeros((n_stop, 2))
    times = np.zeros(n_stop)
    specs = np.zeros(n_stop)
    stokes = np.zeros(n_stop)

    print('before while')
    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (num_spectra < n_stop):
        if msg is not NULL:
            if msg[0].typ in message_types:
                message = message_types[msg[0].typ]
            else:
                message = msg[0].typ
            print(str(' msg type: {0}'.format(message)))

            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                print('valid. num_spectra={0}'.format(num_spectra))
                py_msg = Message.wrap(msg)
                print(str(py_msg))
                bls[num_spectra] = np.asarray(py_msg.info.stations.memview)
                times[num_spectra] = py_msg.info.timestamp/1e6
                specs[num_spectra] = py_msg.info.spectral_window_index
                stokes[num_spectra] = py_msg.info.polarization_product_id
                print('unreffing py_msg')
                py_msg.unref()
                num_spectra += 1
            else:
                print('NULL')
                vysmaw_message_unref(msg)

        print('before msg pop')
        msg = vysmaw_message_queue_timeout_pop(queue, 1000000)
        print('got msg type: {0}'.format(msg[0].typ))

    if handle is not None:
        handle.shutdown()

    return (bls, times, specs, stokes)