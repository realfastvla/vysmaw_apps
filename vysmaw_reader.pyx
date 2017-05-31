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
cdef void filter_none(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
             uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
             void *user_data, bool *pass_filter) nogil:

    cdef unsigned int i

    for i in range(num_infos):
        pass_filter[i] = True

    return


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


cdef class Reader(object):
    """ Object to manage open, read, close for vysmaw application
    """

    cdef double t0
    cdef double t1
    cdef Configuration config
    cdef list consumers
    cdef Handle handle
    cdef unsigned int spec
    cdef unsigned int nspec

#    cdef Consumer c0  # crashes when trying to refer to this object in open
#    cdef vysmaw_message_queue queue0 # crashes when trying to refer to this object in read

    def __cinit__(self, double t0, double t1, str cfile=None):
        """ Open reader with time filter from t0 to t1 in unix seconds
	    cfile is the vys/vysmaw configuration file.
	"""


        self.t0 = t0
        self.t1 = t1
        self.spec = 0

        # configure
        if cfile:
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


    cpdef read(self, nspec):
        """ Read data from consumers
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()
        cdef np.ndarray[np.int_t, ndim=1, mode="c"] typecount = np.zeros(9, dtype=np.int)
        cdef long starttime = time.time()
        cdef long currenttime = starttime
        cdef unsigned int frac
        cdef bool printed = 0

        self.nspec = nspec

        while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (self.spec < self.nspec):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                self.spec += 1
                typecount[msg[0].typ] += 1
                vysmaw_message_unref(msg)
            
            PyErr_CheckSignals()

            currenttime = time.time()
            frac = int(100.*self.spec/self.nspec)
            if not (frac % 25):
                if not printed:
                    print('At spec {0}: {1:1.0f} % of data in {2:1.1f}x realtime'.format(self.spec, 100*float(self.spec)/float(self.nspec), (currenttime-starttime)/(self.t1-self.t0)))
                    printed = 1
            else:
                printed = 0

        print('{0}/{1} spectra received'.format(self.spec, self.nspec))

        return typecount


    cpdef close(self):
        """ Close the vysmaw handle and catch any remaining messages
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()

        if self.handle is not None:
            self.handle.shutdown()

        if self.spec < self.nspec:
            print('Closing vysmaw handle. Remaining messages in queue:')
            while (msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END):
                msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

                if msg is not NULL:
                    print(str('{0}'.format(message_types[msg[0].typ])))
                else:
                    print('NULL')

