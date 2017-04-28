import os.path
import time
import cython
from threading import Thread
from cpython cimport PyErr_CheckSignals
import numpy as np
cimport numpy as np
from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    cdef unsigned int i

    for i in range(num_infos):
        pass_filter[i] = select[0] <= infos[i].timestamp/1e9 and infos[i].timestamp/1e9 < select[1]
    return


def filter1(t0, t1, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1000000, timeout=30, cfile=None):
    """ Read data between unix times t0 and t1.
    Data structure is assumed to be defined by parameters:
    - nant is number of antennas,
    - nspe is number of spectral windows,
    - nchan is number of channels per window,
    - npol is number of stokes parameters, and
    - inttime_micros is integration time in microseconds.
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

    cdef int ni = int((t1-t0)/(inttime_micros/1e6))  # t1, t0 in seconds, i in microsec
    cdef int nbl = nant*(nant-1)/2
    cdef int nchantot = nspw*nchan
    cdef int nspec = ni*nbl*nspw*npol
    cdef long time0 = int(time.time())
    cdef long time1 = time0

    cdef np.ndarray[np.int_t, ndim=2, mode="c"] blarr = np.array([(ind0, ind1) for ind1 in range(nant) for ind0 in range(0,ind1)])
    cdef np.ndarray[np.complex128_t, ndim=4, mode="c"] data = np.zeros(shape=(ni, nbl, nchantot, npol), dtype='complex128')
    timearr = t0+(inttime_micros/1e6)*np.arange(ni)  # using cdef changes result of comparison with msg_time. why?

    print('Expecting {0} integrations and {1} spectra between times {2} and {3} (timeout {4} s)'.format(ni, nspec, t0, t1, timeout))

    # count until total number of spec is received or timeout elapses
    cdef long spec = 0
    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < nspec) and (time1 - time0 < timeout):
        msg = vysmaw_message_queue_timeout_pop(queue0, 2*inttime_micros)

        if msg is not NULL:
#            print(str('msg: type {0}'.format(msg[0].typ)))
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)
                msg_time = py_msg.info.timestamp/1e9
#                print(msg_time)

                iind = np.argmin(np.abs(timearr-msg_time))

                stations = np.array(py_msg.info.stations)
                bind = np.where([np.all(bl == stations) for bl in blarr])[0][0]

                ch0 = nchan*py_msg.info.spectral_window_index # or baseband_index? or baseband_id?
                pind = py_msg.info.polarization_product_id

#                print(iind, bind, ch0, pind)

                data[iind, bind, ch0:ch0+nchan, pind].real = np.array(py_msg.buffer)[::2]
                data[iind, bind, ch0:ch0+nchan, pind].imag = np.array(py_msg.buffer)[1::2]

                spec = spec + 1
                py_msg.unref()
#                print('{0}/{1} spectra received'.format(spec, nspec))
            else:
                vysmaw_message_unref(msg)

        else:
#            print('msg: NULL')
             pass

        time1 = int(time.time())
        PyErr_CheckSignals()

    if handle is not None:
        handle.shutdown()

    return data


def filter2(t0, t1, t2, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1000000, timeout=10, cfile=None):
    """ Read data between two time windows (unix times) t0-t1 and t1-t2.
    Data structure is assumed to be defined by parameters:
    - nant is number of antennas,
    - nspe is number of spectral windows,
    - nchan is number of channels per window,
    - npol is number of stokes parameters, and
    - inttime_micros is integration time in microseconds.
    Will return numpy array when all data collected or as much as is ready when timeout elapses.
    cfile is the vys/vysmaw configuration file.
    """

    # define time window
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] window0 = np.array([t0, t1], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] window1 = np.array([t1, t2], dtype=np.float64)
    cdef int nwindow = 2

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
    cdef void **u = <void **>malloc(nwindow*sizeof(void *))
    u[0] = &window0[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    u[1] = &window1[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

    # set filters
    cdef vysmaw_spectrum_filter *f = \
        <vysmaw_spectrum_filter *>malloc(nwindow*sizeof(vysmaw_spectrum_filter))

    f[0] = filter_time
    f[1] = filter_time
    handle, consumers = config.start(nwindow, f, u)
    free(f)
    free(u)

    cdef Consumer c0 = consumers[0]
    cdef Consumer c1 = consumers[1]
    cdef vysmaw_message_queue queue0 = c0.queue()
    cdef vysmaw_message_queue queue1 = c1.queue()

    threads = []
    t0 = Thread(target=run, args=(queue0, t0, t1, inttime_micros, nant, nspw, nchan, npol, timeout))
    t1 = Thread(target=run, args=(queue1, t1, t2, inttime_micros, nant, nspw, nchan, npol, timeout))
    threads.append(t0)
    threads.append(t1)
    t0.start()
    t1.start()

    if handle is not None:
        handle.shutdown()


cdef void run(vysmaw_message_queue queue, float t0, float t1, long inttime_micros, long nant, long nspw, long nchan, long npol, long timeout):
    return