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

message_types = dict(zip([VYSMAW_MESSAGE_VALID_BUFFER, VYSMAW_MESSAGE_ID_FAILURE, VYSMAW_MESSAGE_QUEUE_ALERT, 
	      VYSMAW_MESSAGE_DATA_BUFFER_STARVATION, VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION, 
	      VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE, VYSMAW_MESSAGE_RDMA_READ_FAILURE, VYSMAW_MESSAGE_VERSION_MISMATCH,
	      VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW, VYSMAW_MESSAGE_END],
	      ["VYSMAW_MESSAGE_VALID_BUFFER", "VYSMAW_MESSAGE_ID_FAILURE", "VYSMAW_MESSAGE_QUEUE_ALERT", 
	      "VYSMAW_MESSAGE_DATA_BUFFER_STARVATION", "VYSMAW_MESSAGE_SIGNAL_BUFFER_STARVATION", 
	      "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE", "VYSMAW_MESSAGE_RDMA_READ_FAILURE", "VYSMAW_MESSAGE_VERSION_MISMATCH",
	      "VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW", "VYSMAW_MESSAGE_END"]))


#cdef class arguments:
#    cdef float t0
#    cdef float t1
#
#    def __cinit__(self, t0, t1):
#        self.t0 = t0
#        self.t1 = t1


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

    cdef np.float64_t t0 = select[0]
    cdef np.float64_t t1 = select[1]
    cdef np.float64_t bbid = select[2]

    for i in range(num_infos):
        ts = infos[i].timestamp/1e9
        if t0 <= ts and ts < t1:
            pass_filter[i] = True
        else:
            pass_filter[i] = False

    return


cdef class Reader(object):
    """ Object to manage open, read, close for vysmaw application
    """

    cdef double t0
    cdef double t1
    cdef int timeout
    cdef Configuration config
    cdef int offset
    cdef list consumers
    cdef Handle handle
    cdef unsigned int spec   # counter of number of spectra received
    cdef unsigned int nspec  # number of spectra expected

#    cdef Consumer c0  # crashes when trying to refer to this object in open
#    cdef vysmaw_message_queue queue0 # crashes when trying to refer to this object in read

    def __cinit__(self, double t0 = 0, double t1 = 0, str cfile = None, int timeout = 10):
        """ Open reader with time filter from t0 to t1 in unix seconds
            If t0/t1 left at default values of 0, then all times accepted.
            cfile is the vys/vysmaw configuration file.
            timeout is time beyond the t1-t0 window.
	"""

        self.t0 = t0
        self.t1 = t1
        self.spec = 0
        self.timeout = timeout

        self.offset = 4  # (integer) seconds early to open handle

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

        cdef long currenttime = time.time()

        if currenttime < self.t0 - self.offset:
            print('Holding for time {0} (less offset {1})'.format(self.t0, self.offset))
            while currenttime < self.t0 - self.offset:
                time.sleep(0.1)
                currenttime = time.time()

        self.open()
        return self


    def __exit__(self, *args):
        self.close()


    cpdef open(self):
        """ Create the handle and consumers
        """

        # define filter inputs
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] window = np.array([self.t0, self.t1], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] bbids = np.array([0], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] filterarr = np.concatenate((window, bbids))

        # set windows
        cdef void **u = <void **>malloc(sizeof(void *))
        u[0] = &filterarr[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
#        u[0] = arguments(self.t0, self.t1)

        # set filters
        cdef vysmaw_spectrum_filter *f = \
            <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

        if self.t0 and self.t1:
            f[0] = filter_time
            self.handle, self.consumers = self.config.start(1, f, u)
        else:
            f[0] = filter_none
            self.handle, self.consumers = self.config.start(1, f, NULL)

        free(f)
        free(u)


    cpdef readwindow(self, antlist, bbsplist, nchan, pollist, inttime_micros):
        """ Read in the time window and place in numpy array of given shape
        antlist is list of antenna numbers (1-based)
        bbsplist is list of "bbid-spwid" and where to place them.
        nchan is number of channels per subband, assumed equal for all subbands received.
        pollist is list polarization indexes and where to place them.
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()

        cdef unsigned int ni = int(round((self.t1-self.t0)/(inttime_micros/1e6)))  # t1, t0 in seconds, i in microsec
        cdef unsigned int nant = len(antlist)
        cdef unsigned int nbl = nant*(nant-1)/2  # cross hands only
        cdef unsigned int nspw = len(bbsplist)
        cdef unsigned int npol = len(pollist)
        cdef unsigned int nchantot = nspw*nchan

        cdef unsigned int frac
        cdef unsigned int bind
        cdef int bind0
        cdef bool printed = 0

        cdef list blarr = ['{0}-{1}'.format(antlist[ind0], antlist[ind1]) for ind1 in range(len(antlist)) for ind0 in range(ind1)]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] timearr = self.t0+(inttime_micros/1e6)*(np.arange(ni)+0.5)
        cdef np.ndarray[np.complex64_t, ndim=4, mode="c"] data = np.zeros(shape=(ni, nbl, nchantot, npol), dtype='complex64')
        self.nspec = ni*nbl*nspw*npol

        cdef long starttime = time.time()
        cdef long currenttime = starttime

        print('Expecting {0} ints, {1} bls, and {2} total spectra between times {3} and {4} (timeout {5:.1f}+{6} s)'.format(ni, nbl, self.nspec, self.t0, self.t1, self.t1-self.t0, self.timeout))
#        print('blarr: {0}. pollist {1}'.format(blarr, pollist))

        if starttime < self.t1 + self.offset:
            # count until total number of spec is received or timeout elapses
            while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (self.spec < self.nspec) and (currenttime - starttime < self.timeout + self.t1-self.t0 + self.offset):
                msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

                if msg is not NULL:
                    if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                        py_msg = Message.wrap(msg)

                        # get the goodies asap
                        msg_time = py_msg.info.timestamp/1e9
                        iind = np.argmin(np.abs(timearr-msg_time))
                        bbid = py_msg.info.baseband_id
                        spid = py_msg.info.spectral_window_index
                        ch0 = nchan*bbsplist.index('{0}-{1}'.format(bbid, spid))

                        # find pol in pollist
                        pind0 = -1
#                        print('polid {0}'.format(py_msg.info.polarization_product_id))
                        for pind in range(len(pollist)):
                            if py_msg.info.polarization_product_id == pollist[pind]:
                                pind0 = pind
                                break

                        # find bl i blarr
                        blstr = '{0}-{1}'.format(py_msg.info.stations[0], py_msg.info.stations[1])
#                        print('blstr: {0} {1}'.format(blstr, type(blstr)))
                        bind0 = -1
                        for bind in range(len(blarr)):
                            if blstr == blarr[bind]:
                                bind0 = bind
                                break

                        # put data in numpy array, if an index exists
                        if bind0 > -1 and pind0 > -1:
#                            print('For iind, bind0, ch0, nchan, pind0: {0}, {1}, {2}, {3}, {4}...'.format(iind, bind0, ch0, nchan, pind0))
#                            print('\tspectrum pointer: {0:x}'.format(<uintptr_t>&py_msg._c_message[0].content.valid_buffer.spectrum))
#                            print('\tbuffer pointer: {0:x}'.format(<uintptr_t>&py_msg._c_message[0].content.valid_buffer.buffer))
                            spectrum = np.array(py_msg.spectrum)  # copy=True is slow. could avoid copy and get pointer (?)
                            data[iind, bind0, ch0:ch0+nchan, pind0].real = spectrum[::2] # slow
                            data[iind, bind0, ch0:ch0+nchan, pind0].imag = spectrum[1::2] # slow
                            self.spec = self.spec + 1
                        else:
                            pass
#                            print('No bind or pind found for {0} {1} {2}'.format(blstr, bind0, pind0))

                        py_msg.unref()  # this may be optional

                    else:
                        print(str('Unexpected message type: {0}'.format(message_types[msg[0].typ])))
                        vysmaw_message_unref(msg)

                currenttime = time.time()
                frac = int(100.*self.spec/self.nspec)
                if not (frac % 20) and frac > 0:
                    if not printed:
                        print('At spec {0}: {1:1.0f}% of data in {2:1.1f}x realtime'.format(self.spec, 100*float(self.spec)/float(self.nspec), (currenttime-starttime)/(self.t1-self.t0)))
                        printed = 1
                else:
                    printed = 0

                PyErr_CheckSignals()

            # after while loop, check reason for ending
            if currenttime-starttime >= self.timeout + self.t1-self.t0 + self.offset:
                print('Reached timeout of {0}s. Exiting...'.format(self.timeout))
            elif msg is not NULL:
                if msg[0].typ is VYSMAW_MESSAGE_END:
                    print('Received VYSMAW_MESSAGE_END. Exiting...')
                
            print('{0}/{1} spectra received'.format(self.spec, self.nspec))

        else:
            print('Start time {0} is later than window end {1} (plus offset {2}). Skipping.'.format(starttime, self.t1, self.offset))

        return data


    cpdef readn(self, nspec):
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

        msgcnt = dict(zip(message_types.keys(), [0]*len(message_types)))

        if self.handle is not None:
            print('Shutting vysmaw down...')
            self.handle.shutdown()

        if self.spec < self.nspec:
            nulls = 0
            while (msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END):
                msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

                if msg is not NULL:
                    msgcnt[msg[0].typ] += 1
#                    print(msg[0].typ)
                else:
                    nulls += 1
#                    print("NULL")

                PyErr_CheckSignals()

            print('Remaining messages in queue: {0}'.format(msgcnt))
            if nulls:
                print('and {0} NULLs'.format(nulls))

