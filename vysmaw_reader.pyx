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
        pass_filter[i] = select[0] <= infos[i].timestamp/1e9 and infos[i].timestamp/1e9 < select[1]

        # test how many times this is called
#        if select[0] <= infos[i].timestamp/1e9 and infos[i].timestamp/1e9 < select[1]:
#            select[2] = select[2] + 1

    return


cdef class Reader(object):
    """ Object to manage open, read, close for vysmaw application
    """

    cdef double t0
    cdef double t1
    cdef Configuration config
    cdef list consumers
    cdef Handle handle

#    cdef Consumer c0  # crashes when trying to refer to this object in open
#    cdef vysmaw_message_queue queue0 # crashes when trying to refer to this object in read

    def __cinit__(self, double t0, double t1, str cfile=None):
        """ Open reader with time filter from t0 to t1 in unix seconds
	    cfile is the vys/vysmaw configuration file.
	"""

        self.t0 = t0
        self.t1 = t1

        # configure
        if cfile:
            assert os.path.exists(cfile), 'Configuration file {0} not found.'.format(cfile)
            print('Reading {0}'.format(cfile))
            self.config = cy_vysmaw.Configuration(cfile)
        else:
            print('Using default vys configuration file')
            self.config = cy_vysmaw.Configuration()


    def __enter__(self):
        """ Context management in Python.
        """

        # **TODO: Could start call to open at self.t0. This might avoid overutilizing resources of the handle.

        self.open()
        return self


    def __exit__(self, *args):
        self.close()


    cpdef open(self):
        """ Create the handle and consumers
        """

        # define time window
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] windows = np.array([self.t0, self.t1], dtype=np.float64)

        # set windows
        cdef void **u = <void **>malloc(sizeof(void *))
        u[0] = &windows[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

        # set filters
        cdef vysmaw_spectrum_filter *f = \
            <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

        f[0] = filter_time
        self.handle, self.consumers = self.config.start(1, f, u)

        free(f)
        free(u)


    cpdef read(self, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1000000, timeout=10, excludeants=[]):
        """ Read data from consumers
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()

        cdef unsigned int ni = int(round((self.t1-self.t0)/(inttime_micros/1e6)))  # t1, t0 in seconds, i in microsec
        cdef unsigned int nbl = nant*(nant-1)/2
        cdef unsigned int nchantot = nspw*nchan
        cdef unsigned int nspec = ni*nbl*nspw*npol
        cdef long starttime = time.time()
        cdef long currenttime = starttime

        cdef np.ndarray[np.int_t, ndim=1, mode="c"] antarr = np.array([ant for ant in np.arange(nant+len(excludeants)) if ant not in excludeants]) # 0 based
#        cdef np.ndarray[np.int_t, ndim=1, mode="c"] antarr = np.array([ant for ant in np.arange(1, nant+1+len(excludeants)) if ant not in excludeants]) # 1 based
        cdef np.ndarray[np.int_t, ndim=2, mode="c"] blarr = np.array([(antarr[ind0], antarr[ind1]) for ind1 in range(len(antarr)) for ind0 in range(0, ind1)])
        cdef np.ndarray[np.complex128_t, ndim=4, mode="c"] data = np.zeros(shape=(ni, nbl, nchantot, npol), dtype='complex128')

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] timearr = self.t0+(inttime_micros/1e6)*(np.arange(ni)+0.5)

        print('Expecting {0} ints, {1} bls, and {2} total spectra between times {3} and {4} (timeout {5} s)'.format(ni, nbl, nspec, self.t0, self.t1, timeout))

        # count until total number of spec is received or timeout elapses
        cdef long spec = 0
        while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < nspec) and (currenttime - starttime < timeout):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                print(str('msg {0}'.format(message_types[msg[0].typ])))
                if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                    py_msg = Message.wrap(msg)

                    # get the goodies asap
                    msg_time = py_msg.info.timestamp/1e9
                    ch0 = nchan*py_msg.info.baseband_index  # TODO: need to be smarter here
                    pind = py_msg.info.polarization_product_id
                    stations = np.array(py_msg.info.stations)
#                    print(msg_time, ch0, pind, stations)

                    # TODO: may be smarter to define acceptable data from input parameters here. drop those that don't fit?
                    hasstations = [np.all(bl == stations) for bl in blarr]
                    if np.any(hasstations):
#                        print('has baseline {0}'.format(stations))

                        bind = np.where(hasstations)[0][0]
                        iind = np.argmin(np.abs(timearr-msg_time))

                        data[iind, bind, ch0:ch0+nchan, pind].real = np.array(py_msg.spectrum)[::2]
                        data[iind, bind, ch0:ch0+nchan, pind].imag = np.array(py_msg.spectrum)[1::2]

                        spec = spec + 1
                    else:
#                        print('no such baseline expected {0}'.format(stations))
                        pass

                    py_msg.unref()

                else:
#                    print('Got an invalid buffer of type {0}'.format(msg[0].typ))
                    vysmaw_message_unref(msg)

            else:
                print('msg: NULL')
#                pass

            currenttime = time.time()
            if currenttime > starttime and spec % 10:
                print('At spec {0}: {1} % of data in {2} % of time'.format(spec, 100*float(spec)/float(nspec), 100*(currenttime-starttime)/(self.t1-self.t0)))

            PyErr_CheckSignals()


        print('{0}/{1} spectra received'.format(spec, nspec))

        return data


    cpdef close(self):
        """ Close the vysmaw handle and catch any remaining messages
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()

        if self.handle is not None:
            self.handle.shutdown()

        print('Closing vysmaw handle. Remaining messages in queue:')
        while (msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)
            if msg is not NULL:
                print(str('msg {0}'.format(message_types[msg[0].typ])))
