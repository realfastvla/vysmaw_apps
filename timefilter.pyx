import time
import cython
from cpython cimport PyErr_CheckSignals
import numpy as np
cimport numpy as np
from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
import cy_vysmaw

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
cdef void filter_time(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef np.float64_t *select = <np.float64_t *>user_data
    cdef unsigned int i

    for i in range(num_infos):
        ts = infos[i].timestamp/1e9
        if select[0] <= ts and ts < select[1]:
            pass_filter[i] = True
        else:
            pass_filter[i] = False

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef filter1(t0, t1, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1000000, timeout=10, cfile=None, excludeants=[]):
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

#    count = 0  # pass this in to count number of callback function calls

    # define time window
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array([t0, t1], dtype=np.float64)

    # configure
    cdef Configuration config
    if cfile:
        print('Reading {0}'.format(cfile))
        config = cy_vysmaw.Configuration(cfile)
    else:
        print('Using default vys configuration file')
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

    cdef unsigned int ni = int(round((t1-t0)/(inttime_micros/1e6)))  # t1, t0 in seconds, i in microsec
    cdef unsigned int nbl = nant*(nant-1)/2
    cdef unsigned int nchantot = nspw*nchan
    cdef unsigned int nspec = ni*nbl*nspw*npol
    cdef long starttime = time.time()
    cdef long currenttime = starttime
    cdef unsigned int frac
    cdef bool printed = 0

    cdef np.ndarray[np.int_t, ndim=1, mode="c"] antarr = np.array([ant for ant in np.arange(1, nant+1+len(excludeants)) if ant not in excludeants]) # 1 based
    cdef list blarr = ['{0}-{1}'.format(antarr[ind0], antarr[ind1]) for ind1 in range(len(antarr)) for ind0 in range(ind1)]
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] timearr = t0+(inttime_micros/1e6)*(np.arange(ni)+0.5)
    cdef np.ndarray[np.complex128_t, ndim=4, mode="c"] data = np.zeros(shape=(ni, nbl, nchantot, npol), dtype='complex128')

    print('Expecting {0} ints, {1} bls, and {2} total spectra between times {3} and {4} (timeout {5} s)'.format(ni, nbl, nspec, t0, t1, timeout))

    # count until total number of spec is received or timeout elapses
    cdef long spec = 0
    while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < nspec) and (currenttime - starttime < timeout):
        msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

        if msg is not NULL:
#            print(str('msg {0}'.format(message_types[msg[0].typ])))
            if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                py_msg = Message.wrap(msg)


                # get the goodies asap
                msg_time = py_msg.info.timestamp/1e9
                ch0 = nchan*py_msg.info.baseband_index  # ** TODO: need to be smarter here
                pind = py_msg.info.polarization_product_id
#                print('ch0, pind: {0}, {1}'.format(ch0, pind))
                blstr = '{0}-{1}'.format(py_msg.info.stations[0], py_msg.info.stations[1])
                spectrum = np.array(py_msg.spectrum, copy=True)  # ** slow, but helps to pull data early and unref
                py_msg.unref()

                # TODO: may be smarter to define acceptable data from input parameters here. drop those that don't fit?
                if blstr in blarr:
#                    print('has baseline {0}'.format(blstr))

                    bind = blarr.index(blstr)
#                    print('bl: {0}, {1}'.format(blstr, bind))
                    iind = np.argmin(np.abs(timearr-msg_time))
#                    print('time: {0}, {1}'.format(msg_time, iind))

                    data[iind, bind, ch0:ch0+nchan, pind].real = spectrum[::2] # slow
                    data[iind, bind, ch0:ch0+nchan, pind].imag = spectrum[1::2] # slow

                    spec = spec + 1

            else:
                print(str('Unexpected message type: {0}'.format(message_types[msg[0].typ])))
                vysmaw_message_unref(msg)
#        else:
#            print('NULL')

        currenttime = time.time()
        frac = int(100.*spec/nspec)
        if not (frac % 25):
            if not printed:
                print('At spec {0}: {1:1.0f} % of data in {2:1.1f}x realtime'.format(spec, 100*float(spec)/float(nspec), (currenttime-starttime)/(t1-t0)))
                printed = 1
        else:
            printed = 0

        PyErr_CheckSignals()

    print('{0}/{1} spectra received'.format(spec, nspec))
#    print('{0} spectra in callback'.format(windows[2]))

    if handle is not None:
        handle.shutdown()

    if spec < nspec:
        msg = NULL
        print('Messages remaining on the queue:')
        while (msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                print(str('{0}'.format(message_types[msg[0].typ])))
            else:
                print('NULL')

    return data
