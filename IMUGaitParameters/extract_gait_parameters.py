from matplotlib import pyplot as pl
from matplotlib.patches import Patch
from scipy import signal
import numpy as np
from numpy import sin, cos, arcsin, arccos, arctan, arctan2
from scipy import stats
import pywt  # pywavelets
from sklearn.decomposition import PCA
import sys
sys.path.append("C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\MC10py")
import MC10py


class GaitParameters:
    """
    Class for storing methods to calculate gait parameters from IMU data.  Minimum necessary IMUs are on both feet,
     both shanks, and the sacrum
    """

    def __init__(self, data, z_thresh=1.25, min_stance_time=0.03, swing_thresh=2, event_match='', alt_still='',
                 turn_split=False, turn_kwargs={}):
        """
        Parameters
        ----------
        data : dict
            Dictionary of data for all subjects.  Structure should be the same as imported using MC10py module.  The
            structure is 'subject->sensor->data type (accel/gyro)->event->raw data'.  This will check for the same
            sensors and events for each subject - if this is NOT the case for any subject that subject will get removed
        z_thresh : float, optional
            Threshold for z values.  Default is 1.25
        min_stance_time : float, optional
            Minimum stance time in seconds for detection.  Defaults to 0.03 seconds.
        swing_thresh : float, int, optional
            Threshold for swing detection in rad/s.  Defaults to 2 rad/s
        event_match : str, optional
            Str to search for in the event type to use only certain events.  Default is '' (no event exclusion)
        alt_still : str, optional
            Alternate event to use for still periods for gyroscope bias removal.  Default is '' (use chosen events)
        turn_split : bool, optional
            Perform turn detection and segmentation.  This will remove time spent turning around from the data.
            Default is False
        turn_kwargs : dict, optional
            Optional turn detection keywords.  Keyword 'plot : bool, optional' determines whether or not data is
            plotted showing detected turns and removed data.
        """

        # TODO update parameter descriptions with more information
        # assign variables to class attributes
        self.data = data
        self.zt = z_thresh
        self.mst = min_stance_time
        self.swt = swing_thresh
        self.astill = alt_still

        self.parameters = ['Step Length', 'Lateral Deviation', 'Step Height', 'Max Swing Velocity', 'Foot Attack Angle',
                           'Contact Time', 'Step Time', 'Cadence']  # list of gait parameters to extract
        self.ev_ord = ['Heavy Back', 'Light Back', 'Heavy Shoulder', 'Light Shoulder']  # order of bags

        _subs = list(self.data.keys())  # initial list of subjects

        # check to make sure all subjects have the required sensors
        _sens_req = ['foot_left', 'foot_right', 'sacrum', 'shank_left', 'shank_right']  # required sensors
        for s in _subs:
            _sens = list(self.data[s].keys())  # get a list of the sensors for the subject
            _sens_good = all([any(i in j for j in _sens) for i in _sens_req])  # check for all required sensors
            if not _sens_good:
                print(f'Removing subject {s} data due to missing required sensors.')
                self.data.pop(s)  # if missing required sensors, remove subject

        _subs = list(self.data.keys())  # update list of subjects
        # get the full names of required sensors
        self.sens = np.array([se for se in self.data[_subs[0]] if any(j in se for j in _sens_req)])

        # check for equal number of events
        _data_check = []
        # keep a record of events for alt still period detection
        for s in _subs:
            # assume all sensors recorded the same events
            _data_check.append(len([e for e in self.data[s][self.sens[0]]['accel'].keys() if event_match in e]))

        self.n_ev = stats.mode(_data_check)[0][0]
        for i in range(len(_data_check)):
            if _data_check[i] < self.n_ev:
                print(f'Removing subject {_subs[i]} data due to missing events.')
                self.data.pop(_subs[i])

        self.subs = list(self.data.keys())  # get final list of subjects

        # get a list of events of interest
        self.events = [e for e in self.data[self.subs[0]][self.sens[0]]['accel'].keys() if event_match in e]

        # get a list of events for alternate still period detection
        if self.astill != '':
            self.alt_events = [e for e in self.data[self.subs[0]][self.sens[0]]['accel'].keys() if self.astill in e]
        else:
            self.alt_events = None

        # convert raw data if it is not in the correct units.  Should be in m/s^2 and rad/s
        # convert the raw acceleration data as part of the _calibration_detect to use the value of gravity
        maxa = []  # get a list of the maximum accelerations for all events
        maxw = []  # get a list of the maximum angular velocities for all events
        for s in self.subs:
            for l in self.data[s].keys():
                for e in self.data[s][l]['gyro'].keys():
                    maxa.append(np.max(self.data[s][l]['accel'][e][:, 1:]))
                    maxw.append(np.max(self.data[s][l]['gyro'][e][:, 1:]))

        if max(maxa) < 15:  # if the maximum over all events is less than 15, convert to m/s^2
            for s in self.subs:
                for l in self.data[s].keys():
                    for e in self.data[s][l]['accel'].keys():
                        self.data[s][l]['accel'][e][:, 1:] *= 9.81

        if max(maxw) > 50:  # if the maximum over all events is greater than 50, convert to rad/s
            for s in self.subs:
                for l in self.data[s].keys():
                    for e in self.data[s][l]['gyro'].keys():
                        self.data[s][l]['gyro'][e][:, 1:] *= np.pi/180

        # ALLOCATE storage for future use
        self.g = {i: {j: None for j in self.data[i].keys()} for i in self.subs}  # standing gravity vector
        self.turns = {i: {j: [] for j in self.events} for i in self.subs}  # turns
        self.stance = {i: dict() for i in self.subs}  # allocate storage for stance indices
        self.swing = {i: dict() for i in self.subs}  # allocate storage for swing indices
        self.step = {i: dict() for i in self.subs}  # allocate storage for step indices
        self.a_n = {i: {j: dict() for j in self.data[i].keys()} for i in self.subs}  # allocate for nagivational acc
        self.v_n = {i: {j: dict() for j in self.data[i].keys()} for i in self.subs}  # allocate for navigational vel
        self.p_n = {i: {j: dict() for j in self.data[i].keys()} for i in self.subs}  # allocate for navigational pos
        self.gait_params = {i: dict() for i in self.subs}  # gait parameters

    def _mean_stillness(self, bias, grav, i, ax, s, sensor, e, nst, plot):
        mag = np.linalg.norm(self.data[s][sensor]['gyro'][e][:, 1:], axis=1)  # ang. vel. magnitude

        # calculate moving mean
        _mm = np.cumsum(mag)
        mm = _mm[nst - 1:]
        mm[1:] = _mm[nst:] - _mm[:-nst]
        mm /= nst

        # determine minimum moving mean location
        mind = np.argmin(mm) + nst
        var = np.var(mag[mind - nst:mind])

        # store bias
        for l in self.data[s].keys():
            bias[l][i, :] = np.median(self.data[s][l]['gyro'][e][mind - nst:mind, 1:], axis=0)
            grav[l][i, :] = np.median(self.data[s][l]['accel'][e][mind - nst:mind, 1:], axis=0)

        if plot:
            ax[i].plot(self.data[s][sensor]['gyro'][e][:, 0], mag, label=r'$||\vec{\omega}||$')
            ax[i].plot(self.data[s][sensor]['gyro'][e][mind - nst:mind, 0], mag[mind - nst:mind], color='r',
                       alpha=0.5, linewidth=5, label=rf'Min. mean, $\sigma^2$={var:.2e}')
            axt = ax[i].twinx()
            axt.semilogy(self.data[s][sensor]['gyro'][e][nst - 1:, 0], mm, color='orange')
            axt.set_ylabel('Mean')

            ax[i].set_title(f'{e}')
            ax[i].legend(loc='best')

        i += 1

        return i, bias

    def _calibration_detect(self, still_time=2, sensor='dorsal_foot_right', plot=False):
        """
        Detect the calibration points in the data using gyroscope data.

        Parameters
        ----------
        still_time : float, int, optional
            Amount of time in seconds to use for stillness detection.  If using an alternate event, this is the time
            that will be used for gyroscope bias elimination.
        sensor : str, optional
            Sensor to use for stillness detection.  Defaults to 'dorsal_foot_right'
        plot : bool, optional
            Plot raw data and chosen still segments.
        """
        if still_time < 0.5 and self.astill is not None:
            print(f'Input still_time of {still_time}s is too low for reliable results.  still_time set to 0.5s')
            still_time = 0.5
        elif still_time > 1.5 and self.astill is None:
            print(f'Input still_time of {still_time}s is possibly too high to find a reliable still perdiod.')

        for s in self.subs:
            fr = 1/(np.mean(np.diff(self.data[s][sensor]['gyro'][self.events[0]][:, 0])))  # time in seconds
            fnyq = fr/2  # nyquist frequency
            n1 = int(round(fr))  # samples in 1 second
            nst = int(round(still_time * n1))  # samples in still_time seconds

            i = 0

            if self.alt_events is not None:
                # allocate storage for bias and gravity in each sensor
                bias = {l: np.zeros((len(self.alt_events), 3)) for l in self.data[s].keys()}
                grav = {l: np.zeros((len(self.alt_events), 3)) for l in self.data[s].keys()}

                if plot:
                    f, ax = pl.subplots(len(self.alt_events), figsize=(14, 8))
                    f.suptitle(f'Subject: {s}')
                else:
                    ax = None
                for e in self.alt_events:
                    i, bias = self._mean_stillness(bias, grav, i, ax, s, sensor, e, nst, plot)

                if plot:
                    f.tight_layout(rect=[0, 0.03, 1, 0.95])
            else:
                # allocate storage for bias and gravity in each sensor
                bias = {l: np.zeros((self.n_ev, 3)) for l in self.data[s].keys()}
                grav = {l: np.zeros((self.n_ev, 3)) for l in self.data[s].keys()}

                if plot:
                    f, ax = pl.subplots(self.n_ev, figsize=(14, 8))
                    f.suptitle(f'Subject: {s}')
                else:
                    ax = None
                for e in self.events:
                    i, bias = self._mean_stillness(bias, grav, i, ax, s, sensor, e, nst, plot)

                if plot:
                    f.tight_layout(rect=[0, 0.03, 1, 0.95])

            # remove bias from all sensors and events
            for l in self.data[s].keys():
                self.g[s][l] = np.mean(grav[l], axis=0)
                for e in self.data[s][l]['gyro'].keys():
                    self.data[s][l]['gyro'][e][:, 1:] -= np.mean(bias[l], axis=0)

        # converting to m/s^2 from g if necessary
        for s in self.subs:
            for l in self.data[s].keys():
                for e in self.events:
                    if np.linalg.norm(self.g[s][l]) < 4.9:
                        self.g[s][l] *= 9.81
                        self.data[s][l]['accel'][e][:, 1:] *= 9.81

    def _turn_detect(self, plot=False):
        """
        Detect 180 degree turns in walking data.  Designed for a walk and turn trial, with one turn in the middle of
        the data
        """
        pl.close('all')  # close all open plots before running

        l = 'sacrum'  # body location to look through
        for s in self.subs:  # iterate through each subject
            if plot:
                f, ax = pl.subplots(self.n_ev, figsize=(14, 9))
                f.suptitle(f'Subject: {s}')

            i = 0
            fs = 1/np.mean(np.diff(self.data[s][l]['accel'][self.events[0]][:, 0]))  # sampling frequency
            fnyq = fs/2  # nyquist frequency
            tb, ta = signal.butter(2, .75/fnyq, 'lowpass')  # filter gyro data to see turns clearly
            for e in self.events:
                # determine index closest to global z axis (direction of gravity).
                iz = np.argmax(abs(self.g[s][l])) + 1  # +1 because first index is time in raw data

                # filter gyroscopic data
                fd = abs(signal.filtfilt(tb, ta, self.data[s][l]['gyro'][e][:, iz]))
                mfd = np.mean(fd)  # mean of filtered data

                nfd = len(fd)  # length of data

                # find peak indices in the filtered data
                _ips = signal.argrelmax(fd, order=63, mode='wrap')
                # exclude any peaks less than 2x the mean of the filtered data
                ip = _ips[0][np.argwhere(fd[_ips] > 2*mfd)].flatten()
                peaks = []  # peaks of turning
                tpts = []  # turn points

                # for each of the detected peaks
                for k in range(len(ip)):
                    # search for points before and after the peak with values less than 125% of the mean
                    i_bef = np.argwhere(fd[:ip[k]] < 1.25*mfd).flatten()
                    i_aft = np.argwhere(fd[ip[k]:] < 1.25*mfd).flatten() + ip[k]

                    # if there are points before and after
                    if i_bef.size > 0 and i_aft.size > 0:
                        peaks.append(ip[k])  # append the peak to the list of peaks
                        tpts.append(i_bef[-1])  # append the points closest to the peak to the list of turn points
                        tpts.append(i_aft[0])
                    # if there are no points before the peak lower than 125% of the mean
                    elif i_bef.size == 0 or i_bef.size < int(fs):
                        peaks.append(ip[k])  # append the peak
                        tpts.append(i_aft[0])  # use the first point after the peak to go below 125% of mean
                    # if there are no points after the peak lower than 125% of the mean
                    elif i_aft.size == 0 or i_aft.size < int(fs):
                        peaks.append(ip[k])  # append the peak
                        tpts.append(i_bef[-1])  # use the last point before the peak to go below 125% of mean

                peaks = np.asarray(peaks)
                tpts = np.asarray(tpts)

                for loc in self.data[s].keys():
                    for imu in ['gyro', 'accel']:
                        if len(tpts) == 2:
                            self.turns[s][e].append((tpts[0], tpts[1]))
                        elif len(tpts) == 3:
                            if tpts[0] < peaks[0]:
                                self.turns[s][e].append((tpts[0], tpts[1]))
                                self.turns[s][e].append((tpts[2], nfd-1))
                            else:
                                self.turns[s][e].append((0, tpts[0]))
                                self.turns[s][e].append((tpts[1], tpts[2]))
                        elif len(tpts) == 4:
                            if tpts[0] > peaks[0] and tpts[0] < peaks[1]:
                                self.turns[s][e].append((0, tpts[0]))
                                self.turns[s][e].append((tpts[1], tpts[2]))
                                self.turns[s][e].append((tpts[3], nfd-1))
                            elif tpts[-1] > peaks[-1]:
                                if tpts[0] < peaks[0]:
                                    self.turns[s][e].append((tpts[0], tpts[1]))
                                    self.turns[s][e].append((tpts[2], tpts[3]))
                                else:
                                    self.turns[s][e].append((0, tpts[0]))
                                    self.turns[s][e].append((tpts[1], tpts[2]))
                                    self.turns[s][e].append((tpts[3], nfd-1))

                if plot:
                    # ax[i].plot(self.data[s][l]['gyro'][e][:, 0], self.data[s][l]['gyro'][e][:, iz])
                    ax[i].plot(self.data[s][l]['gyro'][e][:, 0], fd)
                    ax[i].plot(self.data[s][l]['gyro'][e][peaks, 0], fd[peaks], 'ro')
                    ax[i].plot(self.data[s][l]['gyro'][e][tpts, 0], fd[tpts], 'k+')
                    for p in ['-1', '-2']:
                        ax[i].plot(self.data[s][l]['gyro'][e+p][:, 0], self.data[s][l]['gyro'][e+p][:, iz], color='red',
                                   alpha=0.4, linewidth=2)
                    ax[i].plot()
                    ax[i].set_title(f'{e}')

                i += 1

    def step_detect(self, plot=False):
        """
        Method for detecting steps from foot sensors in walking data
        """
        # pl.close('all')

        sensors = ['proximal_lateral_shank_right', 'proximal_lateral_shank_left']
        # sensors = ['dorsal_foot_right', 'dorsal_foot_left']

        for s in self.subs:
            if plot:
                f, ax = pl.subplots(ncols=self.n_ev, nrows=2, figsize=(16, 8), sharey=True)
                f.suptitle(f'Subject: {s}')
            n = 0  # figure foot side sensor tracking

            # axis of rotation.  Is always 3 for lateral shank sensors
            irot = 3

            for l in sensors:
                m = 0  # figure event number tracking
                if plot:
                    ax[n, m].set_title(f'{l}')
                for e in self.events:
                    fs = 1/np.mean(np.diff(self.data[s][l]['gyro'][e][:, 0]))  # sampling frequency
                    wave = 'mexh'  # mexican hat wavelet

                    # get sensor orientation by looking at if the maximum value is positive or negative
                    max_ind = np.argmax(abs(self.data[s][l]['gyro'][e][:, irot]))
                    orient = self.data[s][l]['gyro'][e][max_ind, irot] > 0

                    # continuous wavelet transform
                    coefs, freq = pywt.cwt(self.data[s][l]['gyro'][e][:, irot], np.arange(1, 17), wave,
                                           sampling_period=1/fs)

                    mc = np.mean(abs(coefs[5, :]))  # get the mean of the 5th wavelet scale
                    if orient:
                        pks, _ = signal.find_peaks(coefs[5, :], height=mc, distance=25)
                    else:
                        pks, _ = signal.find_peaks(-coefs[5, :], height=mc, distance=25)

                    cf = 1 / (np.mean(np.diff(self.data[s][l]['gyro'][e][pks, 0])))  # rough estimate of cadence
                    # get the index of the CWT coefficients with the scale closest to 2x the cadence
                    ic = np.argmin(abs(freq - (2 * cf)))

                    mc = np.mean(abs(coefs[ic, :]))  # get the mean of the coefficients of the selected frequency

                    # search for peaks and troughs in the 2xCadence scale
                    if orient:  # if signal is correctly oriented
                        lmx, _ = signal.find_peaks(coefs[ic, :], height=mc, distance=25)
                        lmn, _ = signal.find_peaks(-coefs[ic, :], distance=25)
                    else:
                        lmx, _ = signal.find_peaks(-coefs[ic, :], height=mc, distance=25)
                        lmn, _ = signal.find_peaks(coefs[ic, :], distance=25)

                    # assume that peaks were picked up correctly, however there are probably more minimums
                    _tr = []
                    for pk in lmx:
                        i_bef = np.argwhere(lmn < pk)  # get indices of mins before the peak
                        i_aft = np.argwhere(lmn > pk)  # get indices of mins after the peak
                        if i_bef.size > 0 and i_aft.size > 0:  # if mins before and after the peak
                            _tr.extend((lmn[i_bef[-1]], lmn[i_aft[0]]))  # append the closest mins
                        elif i_bef.size == 0:
                            _tr.append(lmn[i_aft[0]])
                        elif i_aft.size == 0:
                            _tr.append(lmn[i_bef[-1]])

                    tr = np.unique(_tr)  # ensure there are no duplicates in the array

                    # combine all the extrema into one list
                    _ext = np.append(tr, lmx)  # first step extrema storage
                    _extt = np.append(['min'] * len(tr), ['max'] * len(lmx))  # extrema type storage

                    srt = np.argsort(_ext)  # get the indices of the sorted array
                    ext = _ext[srt]  # sort the extrema array
                    extt = _extt[srt]  # sort the extrema type array

                    # sort into and save the stance, swing, and step indices
                    self.stance[s][e] = []
                    self.swing[s][e] = []
                    self.step[s][e] = []
                    for i in range(len(ext)-2):
                        if extt[i] == 'min' and extt[i+1] == 'max':  # swing is min to min separated by a maximum
                            self.swing[s][e].append((ext[i], ext[i + 2]))  # append the two minimums surrounding the max
                        elif extt[i] == 'min' and extt[i + 1] == 'min':  # stance is min to min with nothing between
                            self.stance[s][e].append((ext[i], ext[i+1]))  # append the two minimums
                    # redo for full steps
                    for i in range(len(ext)-3):
                        # full step sequence is min (foot-strike) -> min (toe-off) -> max -> min (foot-strike)
                        if extt[i] == 'min' and extt[i+1] == 'min' and extt[i+2] == 'max':
                            self.step[s][e].append((ext[i], ext[i+1], ext[i+3]))
                    if plot:
                        line1, = ax[n, m].plot(self.data[s][l]['gyro'][e][:, 0], self.data[s][l]['gyro'][e][:, irot],
                                               label='Raw Data', color='b')
                        ax[n, m].plot(self.data[s][l]['gyro'][e][lmx, 0], self.data[s][l]['gyro'][e][lmx, irot], 'rx')
                        ax[n, m].plot(self.data[s][l]['gyro'][e][tr, 0], self.data[s][l]['gyro'][e][tr, irot], 'kx')

                        for st in self.stance[s][e]:
                            ax[n, m].plot(self.data[s][l]['gyro'][e][st[0]:st[1], 0],
                                          self.data[s][l]['gyro'][e][st[0]:st[1], irot], linewidth=7, alpha=0.4,
                                          color='r')
                        for sw in self.swing[s][e]:
                            ax[n, m].plot(self.data[s][l]['gyro'][e][sw[0]:sw[1], 0],
                                          self.data[s][l]['gyro'][e][sw[0]:sw[1], irot], linewidth=7, alpha=0.4,
                                          color='g')
                    m += 1
                n += 1

            if plot:
                red = Patch(color='r', alpha=0.4, label='Stance')
                green = Patch(color='g', alpha=0.4, label='Swing')
                ax[0, -1].legend(handles=[line1, red, green])

                ax[0, 0].set_ylabel(r'$\omega$ [$\frac{rad}{s}$]', fontsize=15)
                ax[1, 0].set_ylabel(r'$\omega$ [$\frac{rad}{s}$]', fontsize=15)

                f.tight_layout(rect=[0, 0.03, 1, 0.95])  # tight layout except for the figure title
                f.subplots_adjust(wspace=0)  # no width between subplots

    @staticmethod
    def _calculate_stillness_factor(a, w, g, z_thresh):
        """
        Calculate the 'distance from stillness' or stillness factor zed

        Parameters
        ----------
        a : array_like
            N x 3 array of acceleration data
        w : array_like
            N x 3 array of angular velocity data
        g : array_like
            3D vector of gravity.
        z_thresh : float
            Stillness threshold.  Used to determine if sensor is still or not.

        Returns
        -------
        zed : array_like
            N x 1 array of stillness factor
        zind : array_like
            M x 1 array of indices where zed is less than the threshold z_thresh
        """
        za = stats.zscore(a-g)  # z-score of acceleration less standing still gravity vector, Nx3 vector ouput
        zw = stats.zscore(w)  # z-score of gyroscope, bias has already been removed, Nx3 vector output

        zed = np.sqrt(np.sum(zw**2, axis=1) + np.sum(za**2, axis=1))  # sum of squares of z-scores
        zind = abs(zed) < z_thresh

        return zed, zind.flatten()

    @staticmethod
    def _kalman_filter(t, a, w, g, zind):
        """
        Kalman filter setup and operation for pedestrian tracking

        Parameters
        ----------
        t : array_like
            1D array of N timestamps corresponding to acceleration and angular velocity data.
        a : array_like
            Nx3 array of accelerations
        w : array_like
            Nx3 array of angular velocities
        g : array_like
            1D array of gravity determined during still standing
        zind : array_like
            Boolean array indicating if the distance from stillness (zed) is less than the threshold
        """
        gm = np.linalg.norm(g)  # norm of gravity vector
        n, m = a.shape  # get data size

        if abs(a[0, 0]) > gm:
            a0 = a[0, :] * gm/np.linalg.norm(a[0, 0])
        else:
            a0 = a[0, :]
        pitch = -arcsin(a0[0] / gm)
        roll = arctan(a0[1] / a0[2])
        yaw = 0

        n = len(a[:, 0])

        C = np.array([[cos(pitch) * cos(yaw), (sin(roll) * sin(pitch) * cos(yaw)) - (cos(roll) * sin(yaw)),
                       (cos(roll) * sin(pitch) * cos(yaw)) + (sin(roll) * sin(yaw))],
                      [cos(pitch) * sin(yaw), (sin(roll) * sin(pitch) * sin(yaw)) + (cos(roll) * cos(yaw)),
                       (cos(roll) * sin(pitch) * sin(yaw)) - (sin(roll) * cos(yaw))],
                      [-sin(pitch), sin(roll) * cos(pitch), cos(roll) * cos(pitch)]])
        C_prev = C.copy()

        heading = np.full(n, np.nan)
        heading[0] = yaw

        acc_n = np.full((n, 3), np.nan)  # (3, n)
        acc_n[0, :] = C @ a[0, :]

        vel_n = np.full((n, 3), np.nan)  # (3, n)
        vel_n[0, :] = 0

        pos_n = np.full((n, 3), np.nan)  # (3, n)
        pos_n[0, :] = 0

        distance = np.full(n, np.nan)
        distance[0] = 0

        P = np.zeros((9, 9))

        sigma_omega = 1e-2
        sigma_a = 1e-2

        H = np.concatenate((np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)), axis=1)

        sigma_v = 1e-2
        R = np.diag([sigma_v, sigma_v, sigma_v]) ** 2

        for i in range(1, n):
            dt = t[i] - t[i - 1]

            w_mat = np.asarray([[0, -w[i, 2], w[i, 1]],
                                [w[i, 2], 0, -w[i, 0]],
                                [-w[i, 1], w[i, 0], 0]])

            C = np.linalg.lstsq((2 * np.identity(3) - w_mat * dt).T, (C_prev @ (2 * np.identity(3) + w_mat * dt)).T,
                                rcond=None)[0].T

            acc_n[i, :] = 0.5 * (C + C_prev) @ a[i, :]

            vel_n[i, :] = vel_n[i - 1, :] + ((acc_n[i, :] - np.array([0, 0, gm])) + (acc_n[i - 1, :] -
                                                                                     np.array([0, 0, gm]))) * dt / 2
            pos_n[i, :] = pos_n[i - 1, :] + (vel_n[i, :] + vel_n[i - 1, :]) * dt / 2

            S = np.asarray([[0, -acc_n[i, 2], acc_n[i, 1]],
                            [acc_n[i, 2], 0, -acc_n[i, 0]],
                            [-acc_n[i, 1], acc_n[i, 0], 0]])

            n33 = (3, 3)
            _r1 = np.concatenate((np.identity(3), np.zeros(n33), np.zeros(n33)), axis=1)
            _r2 = np.concatenate((np.zeros(n33), np.identity(3), dt * np.identity(3)), axis=1)
            _r3 = np.concatenate((-dt * S, np.zeros(n33), np.identity(3)), axis=1)

            F = np.concatenate((_r1, _r2, _r3), axis=0)

            Q = np.diag(
                np.asarray([sigma_omega, sigma_omega, sigma_omega, 0, 0, 0, sigma_a, sigma_a, sigma_a]) * dt) ** 2

            P = F @ P @ F.T + Q

            if zind[i]:
                K = np.linalg.lstsq((H @ P @ H.T + R).T, (P @ H.T).T, rcond=None)[0].T

                delta_x = K @ vel_n[i, :]

                P = (np.identity(9) - K @ H) @ P

                a_e = delta_x[:3]
                p_e = delta_x[3:6]
                v_e = delta_x[6:]

                ang_mat = -np.asarray([[0, -a_e[2], a_e[1]],
                                       [a_e[2], 0, -a_e[0]],
                                       [-a_e[1], a_e[0], 0]])

                C = np.linalg.lstsq((2 * np.identity(3) - ang_mat).T, (2 * np.identity(3) + ang_mat).T, rcond=None)[
                        0].T @ C

                vel_n[i, :] -= v_e
                pos_n[i, :] -= p_e

            heading[i] = arctan2(C[1, 0], C[0, 0])
            C_prev = C.copy()

            distance[i] = distance[i - 1] + np.sqrt(
                (pos_n[i, 0] - pos_n[i - 1, 0]) ** 2 + (pos_n[i, 1] - pos_n[i - 1, 1]) ** 2)

        return acc_n, vel_n, pos_n, distance

    # -------------------------------------------------------
    # ------- METHODS FOR CALCULATING GAIT PARAMETERS -------
    # -------------------------------------------------------
    @staticmethod
    def Step_Length(pos):
        return pos[-1, 0]

    @staticmethod
    def Lateral_Deviation(pos):
        return np.ptp(pos[:, 1])

    @staticmethod
    def Step_Height(pos):
        return np.max(pos[:, 2])

    @staticmethod
    def Max_Swing_Velocity(vel):
        return np.max(np.sqrt(np.sum(vel**2, axis=0)))

    @staticmethod
    def Foot_Attack_Angle(vel, ind):
        return arctan2(vel[ind, 2], vel[ind, 0]) * 180/np.pi

    @staticmethod
    def Contact_Time(time, start, stop):
        return time[stop] - time[start]

    @staticmethod
    def Step_Time(time, start, stop):
        return time[stop] - time[start]

    @staticmethod
    def Cadence(time, start, stop):
        return 1/(time[stop] - time[start])

    def process_data(self):
        """
        Process the raw data and run it through a Kalman Filter to obtain heading and other signals
        used to calculate gait parameters
        """
        # sensors used
        sensors = ['dorsal_foot_right', 'dorsal_foot_left']

        # TODO change back to all subjects
        for s in self.subs:
            for l in sensors:
                self.gait_params[s][l] = dict()
                for e in self.events:
                    self.gait_params[s][l][e] = {i: [] for i in self.parameters}
                    zed, zind = GaitParameters._calculate_stillness_factor(self.data[s][l]['accel'][e][:, 1:],
                                                                           self.data[s][l]['gyro'][e][:, 1:],
                                                                           self.g[s][l], self.zt)

                    zind[0:2] = True

                    self.a_n[s][l][e], \
                    self.v_n[s][l][e], \
                    self.p_n[s][l][e], _ = GaitParameters._kalman_filter(self.data[s][l]['accel'][e][:, 0],
                                                                         self.data[s][l]['accel'][e][:, 1:],
                                                                         self.data[s][l]['gyro'][e][:, 1:],
                                                                         self.g[s][l], zind)

                    # f, ax = pl.subplots(2, figsize=(14, 9))
                    # ax[1].plot(self.data[s][l]['accel'][e][:, 0], self.p_n[s][l][e])
                    # ax[1].legend(['x', 'y', 'z'])
                    # ax[1].plot(self.data[s][l]['accel'][e][zind, 0], self.p_n[s][l][e][zind, :], 'x')
                    # ax[1].set_title(f'{s}: {l.split("""_""")[-1]} {e}')
                    #
                    # ax[0].plot(self.data[s]['sacrum']['gyro'][e][:, 0], self.data[s]['sacrum']['gyro'][e][:, 2])
                    #
                    # for tu in self.turns[s][e]:
                    #     ax[0].plot(self.data[s]['sacrum']['gyro'][e][tu[0]:tu[1], 0],
                    #                self.data[s]['sacrum']['gyro'][e][tu[0]:tu[1], 2], color='r', alpha=0.4, linewidth=5)
                    #     ax[1].plot(self.data[s][l]['accel'][e][tu[0]:tu[1], 0], self.p_n[s][l][e][tu[0]:tu[1], :],
                    #                color='r', alpha=0.4, linewidth=4)
                    # f.tight_layout()

                    # create a list of indices where a turn is occurring
                    turn_range = np.array([], dtype=int)
                    for tu in self.turns[s][e]:
                        turn_range = np.append(turn_range, np.arange(tu[0], tu[1]))

                    # iterate through the detected steps
                    for st in self.step[s][e]:
                        # if a step starts or ends during a turn, skip it
                        if st[0] in turn_range or st[2] in turn_range:
                            continue
                        t = self.data[s][l]['accel'][e][st[0]:st[2], 0]

                        # rotate so step is in x direction
                        pca = PCA()
                        pca.fit(self.p_n[s][l][e][st[0]:st[2], :2])
                        coef = pca.components_

                        # create arrays
                        pos_r = np.zeros_like(self.p_n[s][l][e][st[0]:st[2], :])
                        vel_r = np.zeros_like(self.p_n[s][l][e][st[0]:st[2], :])

                        # assign rotated value to arrays
                        pos_r[:, :2] = self.p_n[s][l][e][st[0]:st[2], :2] @ coef.T
                        pos_r[:, 2] = self.p_n[s][l][e][st[0]:st[2], 2]

                        vel_r[:, :2] = self.v_n[s][l][e][st[0]:st[2], :2] @ coef.T
                        vel_r[:, 2] = self.v_n[s][l][e][st[0]:st[2], 2]

                        # correct position so x, y starts at origin with minimum z
                        pos_r -= np.append(pos_r[0, :2], min(pos_r[:, 2])).reshape((1, 3)) * \
                                 np.ones((len(pos_r[:, 0]), 1))

                        # ensure that it is in the positive x-direction
                        if pos_r[-1, 0] < 0:
                            # rotate by 180 deg
                            R = np.array([[cos(np.pi), sin(np.pi), 0], [-sin(np.pi), cos(np.pi), 0], [0, 0, 1]])

                            # A'^T = A^T R => A' = R^T A
                            pos_r = (R.T @ pos_r.T).T
                            vel_r = (R.T @ vel_r.T).T

                        self.gait_params[s][l][e]['Step Length'].append(GaitParameters.Step_Length(pos_r))
                        self.gait_params[s][l][e]['Lateral Deviation'].append(GaitParameters.Lateral_Deviation(pos_r))
                        self.gait_params[s][l][e]['Step Height'].append(GaitParameters.Step_Height(pos_r))
                        self.gait_params[s][l][e]['Max Swing Velocity'].append(GaitParameters.Max_Swing_Velocity(vel_r))
                        self.gait_params[s][l][e]['Foot Attack Angle'].append(GaitParameters.Foot_Attack_Angle(vel_r,
                                                                                                               -1))
                        self.gait_params[s][l][e]['Contact Time'].append(GaitParameters.Contact_Time(t, 0, st[1]-st[0]))
                        self.gait_params[s][l][e]['Step Time'].append(GaitParameters.Step_Time(t, 0, -1))
                        self.gait_params[s][l][e]['Cadence'].append(GaitParameters.Cadence(t, 0, -1))

    def export_gait_parameters(self, file):
        """
        Method for exporting results to a csv file.  Specifically for Walk and Turn data, but may potentially used for
        other data if format matches
        """
        hdr = ','.join(['Subject', 'Foot', 'Bag Type'] + self.parameters)  # create header for the file

        fid = open(file, 'w')  # open file to write to

        fid.write(hdr + '\n')  # write the header plus a new line indicator

        # TODO change back to all subjects
        for s in ['1', '2']:  # self.subs:
            for l in self.gait_params[s].keys():
                for e in self.events:
                    f = l.split('_')[-1]  # get foot side, left or right
                    bt = self.ev_ord[int(e.split('-')[0][-1])-1]  # get bag type

                    for i in range(len(self.gait_params[s][l][e][self.parameters[0]])):
                        line = f'{s},{f},{bt}'
                        for p in self.parameters:
                            line += f',{self.gait_params[s][l][e][p][i]}'
                        fid.write(line + '\n')

        fid.close()

    def subject_plots(self):
        for s in self.subs:
            f, ax = pl.subplots(4, figsize=(14, 8), sharex=True)

            stp_len = [self.gait_params[s]['dorsal_foot_right']['Walk and Turn 1']['Step Length'],
                       self.gait_params[s]['dorsal_foot_left']['Walk and Turn 1']['Step Length']]
            ax[0].boxplot(stp_len)

            stp_hgt = [self.gait_params[s]['dorsal_foot_right']['Walk and Turn 1']['Step Height'],
                       self.gait_params[s]['dorsal_foot_left']['Walk and Turn 1']['Step Height']]
            ax[1].boxplot(stp_hgt)

            fca = [self.gait_params[s]['dorsal_foot_right']['Walk and Turn 1']['Foot Attack Angle'],
                   self.gait_params[s]['dorsal_foot_left']['Walk and Turn 1']['Foot Attack Angle']]
            ax[2].boxplot(fca)

            ct = [self.gait_params[s]['dorsal_foot_right']['Walk and Turn 1']['Contact Time'],
                  self.gait_params[s]['dorsal_foot_left']['Walk and Turn 1']['Contact Time']]
            ax[3].boxplot(ct)

    def Run(self, save_loc, still_time=6, turn=True):
        # test = GaitParameters(raw_data, event_match='Walk and Turn', alt_still='Blind Standing')
        self._calibration_detect(still_time=still_time, plot=False)
        if turn:
            self._turn_detect(plot=False)
        self.step_detect(plot=False)
        self.process_data()
        self.export_gait_parameters(save_loc)




