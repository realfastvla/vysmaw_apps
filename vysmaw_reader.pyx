import cython
import cy_vysmaw
import numpy as np
from vysmaw cimport *
from libc.stdint cimport *
from libc.stdlib cimport *
from cy_vysmaw cimport *
from cython.view cimport array as cvarray
from libc.time cimport time_t
from cpython cimport PyErr_CheckSignals
cimport numpy as cnp

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

cdef extern from "unistd.h" nogil:
    void usleep(unsigned int slt)

cdef extern from "math.h" nogil:
    int round(double arg)

cdef extern from "math.h" nogil:
    double fabs(double arg)


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

    cdef cnp.float64_t *select = <cnp.float64_t *>user_data
    cdef unsigned int i

    cdef cnp.float64_t t0 = select[0]
    cdef cnp.float64_t t1 = select[1]
#    cdef long t0
#    cdef long t1
#    t0 = <long> select[0]*1000
#    t1 = <long> select[1]*1000

    for i in range(num_infos):
        ts = infos[i].timestamp * 1./1000000000
        if t0 <= ts and ts < t1:
            pass_filter[i] = True
        else:
            pass_filter[i] = False

    return


@cython.final
cdef class Reader(object):
    """ Object to manage open, read, close for vysmaw application
    """

    cdef cnp.float64_t t0
    cdef cnp.float64_t t1
    cdef cnp.float64_t timeout
    cdef Configuration config
    cdef int offset
    cdef list consumers
    cdef Handle handle
    cdef long currenttime
    cdef int nchan
    cdef float inttime_micros
    cdef int[::1] antlist
    cdef int[::1] pollist
    cdef int[:, ::1] bbsplist
    cdef unsigned int ni
    cdef unsigned int nant
    cdef unsigned int nbl
    cdef unsigned int nspw
    cdef unsigned int npol
    cdef unsigned int nchantot
#    cdef unsigned int spec   # counter of number of spectra received
    cdef unsigned int nspec  # number of spectra expected
    cdef vysmaw_message_type lastmsgtyp

    def __cinit__(self, cnp.float64_t t0, cnp.float64_t t1, int[::1] antlist,
                  int[::1] pollist, int[:,::1] bbsplist,
                  float inttime_micros = 1000000, int nchan = 32,
                  str cfile = None, cnp.float64_t timeout = 10, int offset = 4):
        """ Open reader with time filter from t0 to t1 in unix seconds
            If t0/t1 left at default values of 0, then all times accepted.
            cfile is the vys/vysmaw configuration file.
            timeout is wait time factor that scales time as timeout*(t1-t0).
            offset is time offset to expect vys data from t0 and t1.
            antlist is list of antenna numbers (1-based)
            bbsplist is array of [bbid, spwid] and where to place them.
            nchan is number of channels per subband, assumed equal for all subbands received.
            pollist is list polarization indexes and where to place them.
        """

        # set parameters
        self.t0 = t0
        self.t1 = t1
        self.timeout = timeout
        self.antlist = antlist
        self.bbsplist = bbsplist
        self.nchan = nchan
        self.pollist = pollist
        self.inttime_micros = inttime_micros
        self.offset = offset  # (integer) seconds early to open handle

        # set reference values
        self.ni = round(1000000*(self.t1-self.t0)/self.inttime_micros)
        self.nant = len(self.antlist)
        self.nbl = self.nant*(self.nant-1)/2  # cross hands only
        self.nspw = len(self.bbsplist)
        self.npol = len(self.pollist)
        self.nchantot = self.nspw*self.nchan
        self.nspec = self.ni*self.nbl*self.nspw*self.npol

        # initialize
        self.lastmsgtyp = VYSMAW_MESSAGE_VALID_BUFFER

        # configure
        if cfile:
            print('Reading {0}'.format(cfile))
            self.config = cy_vysmaw.Configuration(cfile)
        else:
            print('Using default vys configuration file')
            self.config = cy_vysmaw.Configuration()

        # modify configuration
        specbytes = vys_max_spectrum_buffer_size(self.nchan, 1)
        self.config.max_spectrum_buffer_size = specbytes
        self.config.spectrum_buffer_pool_size = self.nspec*specbytes
        print('Setting spectrum size to {0} bytes and buffer size to {1} bytes'.format(self.config._c_configuration.max_spectrum_buffer_size, self.config._c_configuration.spectrum_buffer_pool_size))

    def __enter__(self):
        self.currenttime = time(NULL)

        # wait for window before opening consumer
        if self.currenttime < self.t0 - self.offset:
            print('Holding for time {0} (less offset {1})'.format(self.t0, self.offset))
            while self.currenttime < self.t0 - self.offset:
                usleep(100000)
                self.currenttime = time(NULL)
                PyErr_CheckSignals()

        # if too late, bail
        if self.currenttime > self.t1 + self.offset:
            print('Current time {0} is later than window end {1} (plus offset {2}). Skipping.'.format(self.currenttime, self.t1, self.offset))
            return None
        else:
            self.open()

        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.close()

    cpdef open(self):
        """ Create the handle and consumers
        """

        # define filter inputs
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] filterarr = np.array([self.t0, self.t1], dtype=np.float64)
#        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] filterarr = np.concatenate((window, selectpols))

        # set windows
        cdef void **u = <void **>malloc(sizeof(void *))
        u[0] = &filterarr[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC

        # set filters
        cdef vysmaw_spectrum_filter *f = \
            <vysmaw_spectrum_filter *>malloc(sizeof(vysmaw_spectrum_filter))

        f[0] = filter_time
        self.handle, self.consumers = self.config.start(1, f, u)

        free(f)
        free(u)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef readwindow(self):
        """ Read in the time window and place in numpy array of given shape
        """

        cdef Consumer c0 = self.consumers[0]
        cdef vysmaw_message_queue queue0 = c0.queue()
        cdef vysmaw_data_info info
#        cdef uint64_t msg_time
        cdef double msg_time
        cdef unsigned int specbreak = int(0.2*self.nspec)
        cdef int bind0 = -1
        cdef int pind0 = -1
        cdef int iind
        cdef int iind0
        cdef int ch0
        cdef int i = 0
        cdef int[:, ::1] blarr = np.zeros(shape=(self.nbl, 2), dtype=np.int32)
        cdef double[::1] timearr = np.zeros(shape=(self.ni,))
        cdef double[::1] dtimearr = np.zeros(shape=(self.ni,))
        cdef cnp.float32_t[::1] spectrum = np.zeros(shape=(2*self.nchan,), dtype=np.float32)
        cdef cnp.float32_t[:,:,:,::1] datar = np.zeros(shape=(self.ni, self.nbl, self.nchantot, self.npol), dtype=np.float32)
        cdef cnp.float32_t[:,:,:,::1] datai = np.zeros(shape=(self.ni, self.nbl, self.nchantot, self.npol), dtype=np.float32)

        # initialize
        cdef unsigned int spec = 0
        cdef unsigned int speclast = 0
        cdef unsigned int lastints = min(self.ni, 5) # count speclast in last ints
        cdef vysmaw_message *msg = NULL

        print('Expecting {0} ints, {1} bls, and {2} total spectra between times {3} and {4} (timeout {5:.1f} s)'.format(self.ni, self.nbl, self.nspec, self.t0, self.t1, (self.t1-self.t0)*self.timeout))

        # fill arrays
        for iind in range(self.ni):
            timearr[iind] = self.t0+(self.inttime_micros/1000000)*(iind+0.5)

        for ind1 in range(self.nant):
            for ind0 in range(ind1):
                blarr[i, 0] = self.antlist[ind0]
                blarr[i, 1] = self.antlist[ind1]
                i += 1

        cdef long starttime = time(NULL)
        self.currenttime = time(NULL)

        # old way: count until total number of spec is received or timeout elapses
#        while ((msg is NULL) or (self.lastmsgtyp is not VYSMAW_MESSAGE_END)) and (spec < self.nspec) and (self.currenttime - starttime < self.timeout*(self.t1-self.t0) + self.offset):
        # new way: count spec only in last integration
        while ((msg is NULL) or (self.lastmsgtyp is not VYSMAW_MESSAGE_END)) and (speclast < lastints*self.nspec/self.ni) and (self.currenttime - starttime < self.timeout*(self.t1-self.t0) + self.offset):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                self.lastmsgtyp = msg[0].typ
                if msg[0].typ is VYSMAW_MESSAGE_VALID_BUFFER:

                    if not spec % specbreak:
                        print('At spec {0}: {1:1.0f}% of data in {2:1.1f}x realtime'.format(spec, 100*float(spec)/float(self.nspec), (self.currenttime-starttime)/(self.t1-self.t0)))

                    info = msg[0].content.valid_buffer.info

                    # get the goodies asap.
                    # first, find best time bin
                    msg_time = info.timestamp * 1./1000000000

                    for iind in range(self.ni):
                        dtimearr[iind] = fabs(timearr[iind]-msg_time)

                    iind0 = minind(dtimearr, self.ni)

                    # find starting channel for spectrum
                    ch0 = findch0(info.baseband_id, info.spectral_window_index, self.bbsplist, self.nchan, self.nspw)

                    # find pol in pollist
                    pind0 = findpolind(info.polarization_product_id, self.pollist, self.npol)

                    # find bl i blarr
                    bind0 = findblind(info.stations[0], info.stations[1], blarr, self.nbl)

#                    if bind0 == 0 and pind0 == 0 and ch0 == 0:
#                        print(iind0, msg_time-self.t0)

                    # put data in numpy array, if an index exists
                    if bind0 > -1 and pind0 > -1:
                        spectrum = <cnp.float32_t[:2*self.nchan]> msg[0].content.valid_buffer.spectrum

                        for i in range(self.nchan):
                            datar[iind0, bind0, ch0+i, pind0] = spectrum[2*i]
                            datai[iind0, bind0, ch0+i, pind0] = spectrum[2*i+1]

                        spec += 1
                        if iind0 >= self.ni-lastints:
                            speclast += 1
                    else:
                        pass
#                        print('No bind or pind found for {0} {1} {2}'.format(blstr, bind0, pind0))

                else:
                    print(str('Unexpected message type: {0}'.format(message_types[msg[0].typ])))

                vysmaw_message_unref(msg)

            self.currenttime = time(NULL)

            PyErr_CheckSignals()

        # after while loop, check reason for ending
        if self.currenttime-starttime >= self.timeout*(self.t1-self.t0) + self.offset:
            print('Reached timeout of {0:.1f}s. Exiting...'.format(self.timeout*(self.t1-self.t0)))
        elif speclast == lastints*self.nspec/self.ni:
            print('Read all spectra for last {0} integrations. Exiting...'.format(lastints))
        elif msg is not NULL:
            if msg[0].typ is VYSMAW_MESSAGE_END:
                print('Received VYSMAW_MESSAGE_END. Exiting...')

        print('{0}/{1} spectra received'.format(spec, self.nspec))

        data = np.zeros(shape=(self.ni, self.nbl, self.nchantot, self.npol), dtype=np.complex64)
        data.real = np.asarray(datar)
        data.imag = np.asarray(datai)

        return data

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

        nulls = 0
        while (nulls < 10) and (self.lastmsgtyp is not VYSMAW_MESSAGE_END):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                self.lastmsgtyp = msg[0].typ
                vysmaw_message_unref(msg)
                msgcnt[self.lastmsgtyp] += 1
            else:
                nulls += 1

            PyErr_CheckSignals()

        print('Remaining messages in queue: {0}'.format(msgcnt))
        if nulls:
            print('and {0} NULLs'.format(nulls))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int minind(double[:] arr, int ni) nogil:
    cdef int i
    cdef int mini = 0
    cdef double minimum = arr[mini]

    for i in range(1, ni):
        if minimum > arr[i]:
            minimum = arr[i]
            mini = i

    return mini

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int findpolind(int pol, int[::1] polinds, int npol) nogil:
    cdef int ind
    cdef int ind1 = -1

    for ind in range(npol):
        if pol == polinds[ind]:
            ind1 = ind
            break

    return ind1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int findblind(int st0, int st1, int[:, ::1] blarr, int nbl) nogil:
    cdef int bind0 = -1
    cdef int bind
    cdef int bl0
    cdef int bl1

    for bind in range(nbl):
        bl0 = blarr[bind, 0]
        bl1 = blarr[bind, 1]
        if (st0 == bl0) and (st1 == bl1):
            bind0 = bind
            break

    return bind0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int findch0(int bbid, int spid, int[:,::1] bbsplist, int nchan, int nspw) nogil:
    cdef int i
    cdef int ch0

    for i in range(nspw):
        if (bbsplist[i, 0] == bbid) and (bbsplist[i, 1] == spid):
            ch0 = nchan*i
            break

    return ch0
