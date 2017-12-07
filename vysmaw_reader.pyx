import time as pytime
from libc.time cimport time_t
cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

cdef extern from "unistd.h" nogil:
    void sleep(unsigned int slt)

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

    for i in range(num_infos):
        ts = infos[i].timestamp/1e9
        if t0 <= ts and ts < t1:
            pass_filter[i] = True
        else:
            pass_filter[i] = False

    return


@cython.final
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
    cdef long currenttime
    cdef unsigned int nchan
    cdef double inttime_micros
    cdef list antlist
    cdef list pollist
    cdef list bbsplist
    cdef unsigned int ni
    cdef unsigned int nant
    cdef unsigned int nbl
    cdef unsigned int nspw
    cdef unsigned int npol
    cdef unsigned int nchantot
    cdef unsigned int spec   # counter of number of spectra received
    cdef unsigned int nspec  # number of spectra expected

#    cdef Consumer c0  # crashes when trying to refer to this object in open
#    cdef vysmaw_message_queue queue0 # crashes when trying to refer to this object in read

    def __cinit__(self, double t0 = 0, double t1 = 0, antlist = [], pollist = [], bbsplist = [], inttime_micros = 1e6, nchan = 32, str cfile = None, int timeout = 10):
        """ Open reader with time filter from t0 to t1 in unix seconds
            If t0/t1 left at default values of 0, then all times accepted.
            cfile is the vys/vysmaw configuration file.
            timeout is wait time factor that scales time as timeout*(t1-t0).
        """

        self.t0 = t0
        self.t1 = t1
        self.timeout = timeout
        self.antlist = antlist
        self.bbsplist = bbsplist
        self.nchan = nchan
        self.pollist = pollist
        self.inttime_micros = inttime_micros
        self.ni = int(round((self.t1-self.t0)/(inttime_micros/1e6)))  # t1, t0 in seconds, i in microsec
        self.nant = len(self.antlist)
        self.nbl = self.nant*(self.nant-1)/2  # cross hands only
        self.nspw = len(self.bbsplist)
        self.npol = len(self.pollist)
        self.nchantot = self.nspw*self.nchan
        self.nspec = self.ni*self.nbl*self.nspw*self.npol
        self.spec = 0

        self.offset = 4  # (integer) seconds early to open handle

        # configure
        if cfile:
            print('Reading {0}'.format(cfile))
            self.config = cy_vysmaw.Configuration(cfile)
        else:
            print('Using default vys configuration file')
            self.config = cy_vysmaw.Configuration()

        specbytes = self.nchan*16  # complex128 per channel
        self.config._c_configuration.max_spectrum_buffer_size = specbytes
        self.config._c_configuration.spectrum_buffer_pool_size = self.nspec*specbytes
        print('Setting buffer size to {0} bytes and spectrum size to {1} bytes'.format(self.config._c_configuration.max_spectrum_buffer_size, self.config._c_configuration.spectrum_buffer_pool_size))

    def __enter__(self):
        """ Context management in Python.
        """

        self.currenttime = time(NULL)

        if self.currenttime < self.t0 - self.offset:
            print('Holding for time {0} (less offset {1})'.format(self.t0, self.offset))
            while self.currenttime < self.t0 - self.offset:
                sleep(1)
                self.currenttime = time(NULL)

        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    cpdef open(self):
        """ Create the handle and consumers
        """

        # define filter inputs
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] filterarr = np.array([self.t0, self.t1], dtype=np.float64)
#        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] filterarr = np.concatenate((window, bbids))

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef readwindow(self):
        """ Read in the time window and place in numpy array of given shape
        antlist is list of antenna numbers (1-based)
        bbsplist is list of "bbid-spwid" and where to place them.
        nchan is number of channels per subband, assumed equal for all subbands received.
        pollist is list polarization indexes and where to place them.
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()

        cdef unsigned int specbreak = int(0.2*self.nspec)
        cdef unsigned int bind
        cdef int bind0

        cdef list blarr = ['{0}-{1}'.format(self.antlist[ind0], self.antlist[ind1]) for ind1 in range(len(self.antlist)) for ind0 in range(ind1)]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] timearr = self.t0+(self.inttime_micros/1e6)*(np.arange(self.ni)+0.5)
        cdef np.ndarray[np.complex64_t, ndim=4, mode="c"] data = np.zeros(shape=(self.ni, self.nbl, self.nchantot, self.npol), dtype='complex64')
        cdef unsigned int spec = 0

        cdef long starttime = time(NULL)
        self.currenttime = time(NULL)

        print('Expecting {0} ints, {1} bls, and {2} total spectra between times {3} and {4} (timeout {5:.1f} s)'.format(self.ni, self.nbl, self.nspec, self.t0, self.t1, (self.t1-self.t0)*self.timeout))
#        print('blarr: {0}. pollist {1}'.format(blarr, pollist))

        if starttime < self.t1 + self.offset:
            # count until total number of spec is received or timeout elapses
            while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < self.nspec) and (self.currenttime - starttime < self.timeout*(self.t1-self.t0) + self.offset):
                msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

                if msg is not NULL:
                    if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:
                        py_msg = Message.wrap(msg)

                        # get the goodies asap
                        msg_time = py_msg.info.timestamp/1e9
                        iind = np.argmin(np.abs(timearr-msg_time))
                        bbid = py_msg.info.baseband_id
                        spid = py_msg.info.spectral_window_index
                        ch0 = self.nchan*self.bbsplist.index('{0}-{1}'.format(bbid, spid))

                        # find pol in pollist
                        pind0 = -1
#                        print('polid {0}'.format(py_msg.info.polarization_product_id))
                        for pind in range(len(self.pollist)):
                            if py_msg.info.polarization_product_id == self.pollist[pind]:
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
                            data[iind, bind0, ch0:ch0+self.nchan, pind0].real = spectrum[::2] # slow
                            data[iind, bind0, ch0:ch0+self.nchan, pind0].imag = spectrum[1::2] # slow
                            spec += 1
                        else:
                            pass
#                            print('No bind or pind found for {0} {1} {2}'.format(blstr, bind0, pind0))

                        py_msg.unref()  # this may be optional

                    else:
                        print(str('Unexpected message type: {0}'.format(message_types[msg[0].typ])))
                        vysmaw_message_unref(msg)

                self.currenttime = time(NULL)

                if (spec > 0) and not spec % specbreak:
                    print('At spec {0}: {1:1.0f}% of data in {2:1.1f}x realtime'.format(spec, 100*float(spec)/float(self.nspec), (self.currenttime-starttime)/(self.t1-self.t0)))

                PyErr_CheckSignals()

            # after while loop, check reason for ending
            if self.currenttime-starttime >= self.timeout*(self.t1-self.t0) + self.offset:
                print('Reached timeout of {0:.1f}s. Exiting...'.format(self.timeout*(self.t1-self.t0)))
            elif msg is not NULL:
                if msg[0].typ is VYSMAW_MESSAGE_END:
                    print('Received VYSMAW_MESSAGE_END. Exiting...')
                
            self.spec = spec
            print('{0}/{1} spectra received'.format(self.spec, self.nspec))

        else:
            print('Start time {0} is later than window end {1} (plus offset {2}). Skipping.'.format(starttime, self.t1, self.offset))

        return data


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef readn(self, nspec):
        """ Read data from consumers
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()
        cdef np.ndarray[np.int_t, ndim=1, mode="c"] typecount = np.zeros(9, dtype=np.int)
        cdef long starttime = time(NULL)
        cdef unsigned int frac
        cdef bool printed = 0
        cdef unsigned int spec

        self.currenttime = time(NULL)
        self.nspec = nspec

        while ((msg is NULL) or (msg[0].typ is not VYSMAW_MESSAGE_END)) and (spec < self.nspec):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                spec += 1
                typecount[msg[0].typ] += 1
                vysmaw_message_unref(msg)
            
            PyErr_CheckSignals()

            self.currenttime = time(NULL)
            frac = int(100.*spec/self.nspec)
            if not (frac % 25):
                if not printed:
                    print('At spec {0}: {1:1.0f} % of data in {2:1.1f}x realtime'.format(spec, 100*float(spec)/float(self.nspec), (self.currenttime-starttime)/(self.t1-self.t0)))
                    printed = 1
            else:
                printed = 0

        self.spec = spec
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

