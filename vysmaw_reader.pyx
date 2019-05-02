# cython: language_level=2
import cython
from libc.stdint cimport *
from libc.stdlib cimport *
from libc.stdio cimport printf
from libc.time cimport time_t

import numpy as np
cimport numpy as cnp

from vysmaw import cy_vysmaw
from vysmaw.cy_vysmaw cimport *

cdef extern from "time.h" nogil:
    ctypedef int time_tt
    time_tt time(time_t*)

cdef extern from "unistd.h" nogil:
    void usleep(unsigned int slt)

cdef extern from "math.h" nogil:
    int round(double arg)

cdef extern from "math.h" nogil:
    double fabs(double arg)

cdef extern from "math.h" nogil:
    int floor(double arg)

# remove for latest vysmaw
cdef extern from "vysmaw.h" nogil:
    void vysmaw_message_unref(vysmaw_message *arg)

# new style
message_types = dict(zip([VYSMAW_MESSAGE_SPECTRA, VYSMAW_MESSAGE_QUEUE_ALERT,
                          VYSMAW_MESSAGE_SPECTRUM_BUFFER_STARVATION,
                          VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE,
                          VYSMAW_MESSAGE_VERSION_MISMATCH,
                          VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW,
                          VYSMAW_MESSAGE_END],
                         ["VYSMAW_MESSAGE_SPECTRA", "VYSMAW_MESSAGE_QUEUE_ALERT",
                          "VYSMAW_MESSAGE_SPECTRUM_BUFFER_STARVATION",
                          "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE",
                          "VYSMAW_MESSAGE_VERSION_MISMATCH",
                          "VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW",
                          "VYSMAW_MESSAGE_END"]))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_time(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id, uint8_t spw,
                      uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
                      void *user_data, bool *pass_filter) nogil:

    cdef cnp.float64_t *select = <cnp.float64_t *>user_data
    cdef unsigned int i

    cdef cnp.float64_t t0 = select[0]
    cdef cnp.float64_t t1 = select[1]

    for i in range(num_infos):
        ts = infos[i].timestamp * 1./1000000000
        if t0 <= ts and ts < t1:
            pass_filter[i] = True
        else:
            pass_filter[i] = False

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void filter_timeauto(const char *config_id, const uint8_t *stns, uint8_t bb_idx, uint8_t bb_id,
                          uint8_t spw, uint8_t pol, const vys_spectrum_info *infos, uint8_t num_infos,
                          void *user_data, bool *pass_filter) nogil:

    cdef cnp.float64_t *select = <cnp.float64_t *>user_data
    cdef unsigned int i

    cdef cnp.float64_t t0 = select[0]
    cdef cnp.float64_t t1 = select[1]

    for i in range(num_infos):
        ts = infos[i].timestamp * 1./1000000000
        if (t0 <= ts) and (ts < t1) and ((pol == 0) or (pol == 3)) and (stns[0] != stns[1]):
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
    cdef cnp.float64_t data_timeout
    cdef Configuration config
    cdef int offset
    cdef Consumer consumer
    cdef Handle handle
    cdef long currenttime
    cdef int nchan
    cdef float inttime_micros
    cdef int[::1] antlist
    cdef int[:, ::1] bbsplist
    cdef long[::1] pollist
    cdef bool polauto
    cdef unsigned int ni
    cdef unsigned int nant
    cdef unsigned int nbl
    cdef unsigned int nspw
    cdef unsigned int npol
    cdef unsigned int nchantot
    cdef unsigned long nspec  # number of spectra expected
    cdef vysmaw_message_type lastmsgtyp

    def __cinit__(self, cnp.float64_t t0, cnp.float64_t t1, int[::1] antlist,
                  int[:,::1] bbsplist, bool polauto, float inttime_micros=1000000, int nchan=32,
                  str cfile=None, cnp.float64_t timeout=10, int offset=4, 
                  cnp.float64_t data_timeout=3.0):
        """ Open reader with time filter from t0 to t1 in unix seconds
            If t0/t1 left at default values of 0, then all times accepted.
            cfile is the vys/vysmaw configuration file.
            timeout is wait time factor that scales time as timeout*(t1-t0).
            data_timeout is wait time since last spectrum in seconds.
            offset is time offset to expect vys data from t0 and t1.
            antlist is list of antenna numbers (1-based)
            bbsplist is array of [bbid, spwid] and where to place them.
            polauto is bool that defines selecting of only auto pols (rr, ll)
            nchan is number of channels per subband, assumed equal for all subbands received.
        """

        # set parameters
        self.t0 = t0
        self.t1 = t1
        self.timeout = timeout
        self.data_timeout = data_timeout
        self.antlist = antlist
        self.bbsplist = bbsplist
        self.polauto = polauto
        self.nchan = nchan
        self.inttime_micros = inttime_micros
        self.offset = offset  # (integer) seconds early to open handle

        # set reference values
        self.ni = round(1000000*(self.t1-self.t0)/self.inttime_micros)
        self.nant = len(self.antlist)
        self.nbl = self.nant*(self.nant-1)/2  # cross hands only
        self.nspw = len(self.bbsplist)
        if self.polauto:
            self.pollist = np.array([0, 3])
        else:
            self.pollist = np.array([0, 1, 2, 3])
        self.npol = len(self.pollist)
        self.nchantot = self.nspw*self.nchan
        self.nspec = self.ni*self.nbl*self.nspw*self.npol

        # initialize
        self.lastmsgtyp = VYSMAW_MESSAGE_SPECTRA

        # configure
        if cfile:
            print('Reading {0}'.format(cfile))
            self.config = cy_vysmaw.Configuration(cfile)
        else:
            print('Using default vys configuration file')
            self.config = cy_vysmaw.Configuration()

        # modify configuration
        specbytes = 8*self.nchan
        self.config.max_spectrum_buffer_size = specbytes
        self.config.spectrum_buffer_pool_size = self.nspec*specbytes
        print('Setting spectrum size to {0} bytes and buffer size to {1} bytes'.format(self.config._c_configuration.max_spectrum_buffer_size, self.config._c_configuration.spectrum_buffer_pool_size))
        self.currenttime = time(NULL)

    def __enter__(self):
        self.currenttime = time(NULL)

        # wait for window before opening consumer
        if self.currenttime < self.t0 - self.offset:
            print('Holding for time {0} (less offset {1})'.format(self.t0, self.offset))
            with nogil:
                while self.currenttime < self.t0 - self.offset:
                    usleep(100000)
                    self.currenttime = time(NULL)
#                    PyErr_CheckSignals()

        # if too late, bail
        if self.currenttime > self.t1 + self.offset:
            print('Current time {0} is later than window end {1} (plus offset {2}). Skipping.'.format(self.currenttime, self.t1, self.offset))
            return None
        else:
            print('Setting up vysmaw filter to start consumer')
            self.open()

        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.close()

    cdef open(self):
        """ Create the handle and consumer
        """

# old way
#        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] filterarr = np.array([self.t0, self.t1], dtype=np.float64)
#        cdef void **u = <void **>malloc(sizeof(void *))
#        u[0] = &filterarr[0]       # See https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
#        self.handle, self.consumer = self.config.start(f, u)
#        free(u)

        cdef double[::1] filterarr_memview
        filterarr_memview = np.ascontiguousarray([self.t0, self.t1])

        if self.polauto:
            f = filter_timeauto
        else:
            f = filter_time

        printf('Creating vysmaw handle and consumer\n')
        self.handle, self.consumer = self.config.start(f, &filterarr_memview[0])

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef readwindow(self):
        """ Read in the time window and place in numpy array of given shape
        """

        if self.handle is None:
            print("vysmaw handle not set. Be sure to use in 'with' context.")
            raise SystemExit

        cdef Consumer c0 = self.consumer
        cdef vysmaw_message_queue queue0 = c0.queue()
        cdef vysmaw_data_info info
        cdef double msg_time = 0.0
        cdef unsigned long specbreak = int(0.2*self.nspec)
        cdef int bind0
        cdef int pind0
        cdef int iind0
        cdef int ch0
        cdef int i = 0
        cdef int j = 0
        cdef unsigned int max_nant = 29
        cdef int[:, ::1] blarr = np.zeros(shape=(max_nant, max_nant), dtype=np.int32)
        cdef cnp.complex64_t[:, :, :, ::1] data = np.zeros(shape=(self.ni, self.nbl, self.nchantot, self.npol), dtype=np.complex64)

        # initialize
        cdef unsigned long spec = 0
        cdef unsigned long spec_good = 0
        cdef unsigned long spec_dup = 0
        cdef unsigned long spec_out = 0
        cdef unsigned long spec_invalid = 0
        cdef unsigned long spec_badbl = 0
        cdef unsigned long spec_badch = 0
        cdef unsigned long spec_badpol = 0
        cdef int[::1] msgcnt = np.zeros(shape=(len(message_types),), dtype=np.int32)
        cdef float readfrac
        cdef float rtfrac
        cdef float latency
        cdef unsigned long speclast = 0
        cdef unsigned int lastints = min(self.ni, 10) # count speclast in last ints
        cdef vysmaw_message *msg = NULL

        print('Expecting {0} ints, {1} bls, and {2} total spectra of length {3} between times {4} and {5} (timeout {6:.1f} s)'.format(self.ni, self.nbl, self.nspec, self.nchan, self.t0, self.t1, (self.t1-self.t0)*self.timeout))

        for ind1 in range(max_nant):
            for ind0 in range(max_nant):
                blarr[ind0, ind1] = -1

        i = 0
        for ind1 in range(self.nant):
            for ind0 in range(ind1):
                blarr[self.antlist[ind0], self.antlist[ind1]] = i
                i += 1

        cdef long starttime = time(NULL)
        cdef long lasttime = 0

        while (msg is NULL) or (self.lastmsgtyp is not VYSMAW_MESSAGE_END):
            with nogil:
                msg = vysmaw_message_queue_timeout_pop(queue0, 100000)
                self.currenttime = time(NULL)

                if msg is not NULL:
                    self.lastmsgtyp = msg[0].typ
                    lasttime = self.currenttime
                    if msg[0].typ is VYSMAW_MESSAGE_SPECTRA:
                        info = msg[0].content.spectra.info
                        spec += msg[0].content.spectra.num_spectra

                        # get the goodies asap.
                        # find starting channel for spectrum
                        ch0 = findch0(info.baseband_id, info.spectral_window_index, self.bbsplist, self.nchan, self.nspw)

                        # find pol in pollist
                        pind0 = findpolind(info.polarization_product_id, self.pollist, self.npol)

                        # find bl i blarr
                        bind0 = blarr[info.stations[0], info.stations[1]]

                        # put data in numpy array, if an index exists
                        if bind0 > -1 and ch0 > -1 and pind0 > -1:
                            specpermsg = msg[0].content.spectra.num_spectra
                            for i in range(specpermsg):
                                # rdma_read_status requires GIL for now
#                                if (msg[0].data[i].failed_verification is False) and (msg[0].data[i].rdma_read_status == b""):
                                if (msg[0].data[i].failed_verification is False) and (msg[0].data[i].values is not NULL):
                                    msg_time = msg[0].data[i].timestamp * 1./1000000000
                                    iind0 = (int)((msg_time-self.t0)/(self.inttime_micros*1e-6))

                                    if not spec_good % specbreak:
                                        readfrac = 100.*spec_good * 1./self.nspec
                                        rtfrac = (self.currenttime-starttime)/(self.t1-self.t0)
                                        latency = self.currenttime - msg_time
                                        printf('At spec %lu: %1.0f%% of data in %1.1fx realtime (latency=%.3fs)\n', spec_good, readfrac, rtfrac, latency)

                                    if iind0<0 or iind0>=self.ni:
                                        # Out of time range
                                        spec_out += 1
                                    elif data[iind0, bind0, ch0, pind0] != 0j:
                                        #printf('Already set index: %d %d %d %d\t', iind0, bind0, ch0, pind0)
                                        spec_dup += 1
                                    else:
                                        spec_good += 1
                                        for j in range(self.nchan):
                                            data[iind0, bind0, ch0+j, pind0] = msg[0].data[i].values[j]
                                        if iind0 >= self.ni-lastints:
                                            speclast += 1
                                else:
                                    #printf('Invalid spectrum\t')
                                    spec_invalid += 1

                        elif bind0 == -1:
                            #printf('bind not found for (%d, %d)\t', info.stations[0], info.stations[1])
                            spec_badbl += msg[0].content.spectra.num_spectra
                        elif ch0 == -1:
                            #printf('ch0 not found for (bbid, spwid) = (%d, %d)\t', info.baseband_id, info.spectral_window_index)
                            spec_badch += msg[0].content.spectra.num_spectra
                        elif pind0 == -1:
                            #printf('pind not found for %d\t', info.polarization_product_id)
                            spec_badpol += msg[0].content.spectra.num_spectra

                    else:
#                        msgcnt[int(msg[0].typ)] += 1
                        msgcnt[int(self.lastmsgtyp)] += 1

                    vysmaw_message_unref(msg)

            # Check for a ending condition
            if self.currenttime-starttime >= self.timeout*(self.t1-self.t0) + self.offset:
                print('Reached timeout of {0:.1f}s. Exiting...'.format(self.timeout*(self.t1-self.t0)))
                break
            elif (lasttime>0) and (self.currenttime-lasttime>self.data_timeout):
                print('No data for last {0:.1f}s. Exiting...'.format(self.data_timeout))
                break
            elif speclast == lastints*self.nspec/self.ni:
                print('Read all spectra for last {0} integrations. Exiting...'.format(lastints))
                break
            elif self.lastmsgtyp is VYSMAW_MESSAGE_END:
                print('Received VYSMAW_MESSAGE_END. Exiting...')
                break

        # Report stats
        #print('{0}/{1} spectra received'.format(spec, self.nspec))
        print('Total spectra Exepcted: {0}'.format(self.nspec))
        print('              Received: {0} ({1:.3f}%)'.format(spec,100.0*float(spec)/float(self.nspec)))
        print('          Good spectra: {0} ({1:.3f}%)'.format(spec_good,100.0*float(spec_good)/float(self.nspec)))
        print('            Duplicates: {0}'.format(spec_dup))
        print('     Out of time range: {0}'.format(spec_out))
        print('               Invalid: {0}'.format(spec_invalid))
        print('                Bad bl: {0}'.format(spec_badbl))
        print('              Bad chan: {0}'.format(spec_badch))
        print('               Bad pol: {0}'.format(spec_badpol))
        for i in range(len(message_types)):
            if msgcnt[i] > 0:
                print('vysmaw message type {0}: {1}'.format(message_types[i], msgcnt[i]))

        if spec_good > 0:
            return np.asarray(data)
        else:
            return None

    cdef close(self):
        """ Close the vysmaw handle and catch any remaining messages
        """

        cdef vysmaw_message *msg = NULL
        cdef Consumer c0 = self.consumer
        cdef vysmaw_message_queue queue0 = c0.queue()

        msgcnt = dict(zip(message_types.keys(), [0]*len(message_types)))

        if self.handle is not None:
            print('Shutting vysmaw down...')
            self.handle.shutdown()

        nulls = 0
        while (nulls < 20) and (self.lastmsgtyp is not VYSMAW_MESSAGE_END):
            msg = vysmaw_message_queue_timeout_pop(queue0, 100000)

            if msg is not NULL:
                self.lastmsgtyp = msg[0].typ
                vysmaw_message_unref(msg)
                msgcnt[self.lastmsgtyp] += 1
            else:
                nulls += 1

#            PyErr_CheckSignals()

        print('Remaining messages in queue: {0}'.format(msgcnt))
        if nulls:
            print('and {0} NULLs'.format(nulls))


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int findpolind(int pol, long[::1] polinds, int npol) nogil:
    cdef int ind
    cdef int ind1 = -1

    for ind in range(npol):
        if pol == polinds[ind]:
            ind1 = ind
            break

    return ind1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int findch0(int bbid, int spid, int[:, ::1] bbsplist, int nchan, int nspw) nogil:
    cdef int i
    cdef int ch0 = -1

    for i in range(nspw):
        if (bbsplist[i, 0] == bbid) and (bbsplist[i, 1] == spid):
            ch0 = nchan*i
            break

    return ch0
