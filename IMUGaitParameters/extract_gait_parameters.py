from matplotlib import pyplot as pl
from scipy import signal
import numpy as np
from numpy import sin, cos, arcsin, arccos, arctan, arctan2
from scipy import stats
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
            Threshold for swing detection
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
        self.raw_data = data
        self.zt = z_thresh
        self.mst = min_stance_time
        self.swt = swing_thresh
        self.astill = alt_still

        _subs = list(self.raw_data.keys())  # initial list of subjects

        # check to make sure all subjects have the required sensors
        _sens_req = ['foot_left', 'foot_right', 'sacrum', 'shank_left', 'shank_right']  # required sensors
        for s in _subs:
            _sens = list(self.raw_data[s].keys())  # get a list of the sensors for the subject
            _sens_good = all([any(i in j for j in _sens) for i in _sens_req])  # check for all required sensors
            if not _sens_good:
                print(f'Removing subject {s} data due to missing required sensors.')
                self.raw_data.pop(s)  # if missing required sensors, remove subject

        _subs = list(self.raw_data.keys())  # update list of subjects
        # get the full names of required sensors
        self.sens = np.array([se for se in self.raw_data[_subs[0]] if any(j in se for j in _sens_req)])

        # check for equal number of events
        _data_check = []
        # keep a record of events for alt still period detection
        for s in _subs:
            # assume all sensors recorded the same events
            _data_check.append(len([e for e in self.raw_data[s][self.sens[0]]['accel'].keys() if event_match in e]))

        self.n_ev = stats.mode(_data_check)[0][0]
        for i in range(len(_data_check)):
            if _data_check[i] < self.n_ev:
                print(f'Removing subject {_subs[i]} data due to missing events.')
                self.raw_data.pop(_subs[i])

        self.subs = list(self.raw_data.keys())  # get final list of subjects

        # get a list of events of interest
        self.events = [e for e in self.raw_data[self.subs[0]][self.sens[0]]['accel'].keys() if event_match in e]

        # get a list of events for alternate still period detection
        if self.astill != '':
            self.alt_events = [e for e in self.raw_data[self.subs[0]][self.sens[0]]['accel'].keys() if self.astill in e]
        else:
            self.alt_events = None

        # PRE-ALLOCATE storage for future use
        self.g = {i: {j: None for j in self.raw_data[i].keys()} for i in self.subs}  # standing gravity vector
        # self.data = {i: dict() for i in self.subs}  # pre-allocate storage for data
        self.gait_params = {i: dict() for i in self.subs}  # pre-allocate storage for gait parameters

    def _mean_stillness(self, bias, grav, i, ax, s, sensor, e, nst, plot):
        mag = np.linalg.norm(self.raw_data[s][sensor]['gyro'][e][:, 1:], axis=1)  # ang. vel. magnitude

        # calculate moving mean
        _mm = np.cumsum(mag)
        mm = _mm[nst - 1:]
        mm[1:] = _mm[nst:] - _mm[:-nst]
        mm /= nst

        # determine minimum moving mean location
        mind = np.argmin(mm) + nst
        var = np.var(mag[mind - nst:mind])

        # store bias
        for l in self.raw_data[s].keys():
            bias[l][i, :] = np.median(self.raw_data[s][l]['gyro'][e][mind - nst:mind, 1:], axis=0)
            grav[l][i, :] = np.median(self.raw_data[s][l]['accel'][e][mind - nst:mind, 1:], axis=0)

        if plot:
            ax[i].plot(self.raw_data[s][sensor]['gyro'][e][:, 0], mag, label=r'$||\vec{\omega}||$')
            ax[i].plot(self.raw_data[s][sensor]['gyro'][e][mind - nst:mind, 0], mag[mind - nst:mind], color='r',
                       alpha=0.5, linewidth=5, label=rf'Min. mean, $\sigma^2$={var:.3f}')
            axt = ax[i].twinx()
            axt.semilogy(self.raw_data[s][sensor]['gyro'][e][nst - 1:, 0], mm, color='orange')
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
            fr = 1/(np.mean(np.diff(self.raw_data[s][sensor]['gyro'][self.events[0]][:, 0])))  # time in seconds
            fnyq = fr/2  # nyquist frequency
            n1 = int(round(fr))  # samples in 1 second
            nst = int(round(still_time * n1))  # samples in still_time seconds

            i = 0

            if self.alt_events is not None:
                # allocate storage for bias and gravity in each sensor
                bias = {l: np.zeros((len(self.alt_events), 3)) for l in self.raw_data[s].keys()}
                grav = {l: np.zeros((len(self.alt_events), 3)) for l in self.raw_data[s].keys()}

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
                bias = {l: np.zeros((self.n_ev, 3)) for l in self.raw_data[s].keys()}
                grav = {l: np.zeros((len(self.alt_events), 3)) for l in self.raw_data[s].keys()}

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
            for l in self.raw_data[s].keys():
                self.g[s][l] = np.mean(grav[l], axis=0)
                for e in self.raw_data[s][l]['gyro'].keys():
                    self.raw_data[s][l]['gyro'][e][:, 1:] -= np.mean(bias[l], axis=0)

    def _turn_detect(self, plot=False):
        """
        Detect 180 degree turns in walking data
        """
        pl.close('all')  # close all open plots before running

        events = self.events.copy()
        l = 'sacrum'  # body location to look through
        for s in ['1', '2']:  # self.subs:  # iterate through each subject
            if plot:
                f, ax = pl.subplots(self.n_ev, figsize=(14, 9))

            i = 0
            fs = 1/np.mean(np.diff(self.raw_data[s][l]['accel'][self.events[0]][:, 0]))  # sampling frequency
            fnyq = fs/2  # nyquist frequency
            tb, ta = signal.butter(1, 1.5/fnyq, 'lowpass')  # filter gyro data to see turns clearly
            for e in self.events:
                # determine index closest to global z axis (direction of gravity).
                iz = np.argmax(self.g[s][l])

                # filter gyroscopic data
                fd = abs(signal.filtfilt(tb, ta, self.raw_data[s][l]['gyro'][e][:, iz]))
                mfd = np.mean(fd)  # mean of filtered data

                # find peak indices in the filtered data
                _ips = signal.argrelmax(fd, order=63, mode='wrap')
                # exclude any peaks less than 2x the mean of the filtered data
                ip = _ips[0][np.argwhere(fd[_ips] > 2*mfd)].flatten()
                turns = []
                for k in range(len(ip)):
                    i_bef = np.argwhere(fd[:ip[k]] < 1.25*mfd).flatten()
                    i_aft = np.argwhere(fd[ip[k]:] < 1.25*mfd).flatten()

                    if i_bef.size > 0 and i_aft.size > 0:
                        turns.append((i_bef[-1], ip[k], i_aft[0] + ip[k]))
                    elif i_bef.size == 0:
                        turns.append((0, ip[k], i_aft[0] + ip[k]))
                    elif i_aft.size == 0:
                        turns.append((i_bef[-1], ip[k], len(fd)-1))

                if plot:
                    i1 = np.array([j[slice(0, 3, 2)] for j in turns]).flatten()
                    i2 = np.array([j[1] for j in turns])
                    # ax[i].plot(self.raw_data[s][l]['gyro'][e][:, 0], self.raw_data[s][l]['gyro'][e][:, iz])
                    ax[i].plot(self.raw_data[s][l]['gyro'][e][:, 0], fd)
                    ax[i].plot(self.raw_data[s][l]['gyro'][e][i2, 0], fd[i2], 'ro')
                    ax[i].plot(self.raw_data[s][l]['gyro'][e][i1, 0], fd[i1], 'k+')
                    ax[i].set_title(f'{e}')

                i += 1

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
    def _kalman_filter(t, a, w, g, zind, min_stance_time, swing_thresh):
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
        min_stance_time : float
            Minimum time required for a stance to have/be occurring
        swing_thresh : float
            Minimum angular velocity value to be considered in the swing phase
        """

        # TODO add inputs to effect sigma and other variables set down below
        fs = 1 / np.mean(np.diff(t))

        n, m = a.shape  # get data size

        state = np.ones(n)  # pre-allocate for state storage
        state[0] = 0
        statef = np.copy(state)  # pre-allocate for smoothed state

        # calculate Euler angles assuming stationary sensor
        pitch = np.mean(-arcsin(a[0, 0] / np.linalg.norm(g)))
        roll = np.mean(arctan(a[0, 1] / a[0, 2]))
        yaw = 0

        # Rotation matrix?
        C = np.array([[cos(pitch), sin(roll) * sin(pitch), cos(roll) * cos(pitch)],
                      [0, cos(roll), -sin(roll)],
                      [-sin(pitch), sin(roll) * cos(pitch), cos(roll) * cos(pitch)]])

        C_prev = C.copy()

        # pre-allocate storage for heading direction.  Indicates direction subject is facing, and not
        # necessarily the direction of travel
        heading = np.zeros(n)
        heading[0] = yaw

        # pre-allocate storage for acceleration in navigational frame
        acc_n = np.zeros((3, n))
        acc_n[:, 0] = C @ a[0, :]

        # pre-allocate storage for velocity in navigational frame, with initial velocity assumed to be 0
        vel_n = np.zeros((3, n))
        vel_n[:, 0] = np.array([0, 0, 0])

        # pre-allocate storage for position in navigational frame.  Initial position arbitrarily set to 0
        pos_n = np.zeros((3, n))
        pos_n[:, 0] = np.array([0, 0, 0])

        # pre-allocate storage for distance traveled used for altitude plots
        distance = np.zeros(n)
        distance[0] = 0

        # Error covariance matrix
        P = np.zeros((9, 9))

        # process noise parameter, gyroscope and accelerometer noise
        sigma_omega = 1e-2
        sigma_a = 1e-2

        # ZUPT measurement matrix
        H = np.concatenate((np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)), axis=1)

        # ZUPT measurement noise covariance matrix
        sigma_v = 1e-3
        R = np.diag([sigma_v] * 3) ** 2

        wmax = 0
        start_swing = 0
        start_stance = 0

        # Main loop for Kalman Filter
        for i in range(1, n):
            # START INS (transformation, double integration)
            dt = t[i] - t[i-1]  # seconds

            # skew symmetric matrix for angular rates
            wm = np.array([[0, -w[i, 2], w[i, 1]],
                           [w[i, 2], 0, -w[i, 0]],
                           [-w[i, 1], w[i, 0], 0]])

            # TODO check lstsq result
            # orientation estimation
            C = C_prev @ np.linalg.lstsq(2 * np.identity(3) + wm * dt, 2 * np.identity(3) - wm * dt, rcond=None)[0]

            # transform the acceleration into the navigational frame
            acc_n[:, i] = 0.5 * (C + C_prev) @ a[i, :]

            # velocity and position using trapezoidal integration
            gz = np.array([0, 0, np.linalg.norm(g)])
            vel_n[:, i] = vel_n[:, i - 1] + ((acc_n[:, i] - gz) + (acc_n[:, i - 1] - gz)) * dt / 2
            pos_n[:, i] = pos_n[:, i - 1] + (vel_n[:, i] + vel_n[:, i - 1]) * dt / 2

            # skew symmetric matrix from navigational frame acceleration
            S = np.array([[0, -acc_n[2, i], acc_n[1, i]],
                          [acc_n[2, i], 0, -acc_n[0, i]],
                          [-acc_n[1, i], acc_n[0, i], 0]])

            # create state transition matrix
            _r1 = np.concatenate((np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))), axis=1)
            _r2 = np.concatenate((np.zeros((3, 3)), np.identity(3), dt * np.identity(3)), axis=1)
            _r3 = np.concatenate((-dt * S, np.zeros((3, 3)), np.identity(3)), axis=1)

            F = np.concatenate((_r1, _r2, _r3), axis=0)

            # compute the process noise covariance
            Q = np.diag([sigma_omega, sigma_omega, sigma_omega, 0, 0, 0, sigma_a, sigma_a, sigma_a]) ** 2

            # propagate the error covariance matrix
            P = F @ P @ np.linalg.inv(F) + Q

            # END of INS

            # stance phase detection and zero-velocity updates
            if zind[i]:
                # START Kalman Filter zero-velocity updates
                state[i] = 0  # stance

                # TODO check lstsq result
                # Kalman Gain
                K, _, _, _ = np.linalg.lstsq((P @ H.transpose()).T, H @ P @ H.transpose() + R, rcond=None)

                # update the filter state
                delta_x = K @ vel_n[:, i]

                # update the error covariance matrix
                P = (np.identity(9) - K @ H) @ P  # simplified covariance update found in most books

                # extract errors from the KF states
                attitude_e = delta_x[:3]
                pos_e = delta_x[3:6]
                vel_e = delta_x[6:]
                # END Kalman Filter zero-velocity update

                # APPLY corrections to INS estimates
                # skew symmetric matrix for small angles to correct orientation
                ang_mat = -np.array([[0, -attitude_e[2], attitude_e[1]],
                                     [attitude_e[2], 0, -attitude_e[0]],
                                     [-attitude_e[1], attitude_e[0], 0]])

                # TODO check lstsq result
                # correct orientation
                C = np.linalg.lstsq(2 * np.identity(3) + ang_mat, 2 * np.identity(3) - ang_mat, rcond=None)[0] @ C

                # correct position and velocity estimates
                vel_n[:, i] -= vel_e
                pos_n[:, i] -= pos_e

            # smooth states
            statef[i] = state[i]

            if state[i] - state[i - 1] == 1:  # 0 (stance) -> 1 (swing)
                # deal with false positive stance
                if (i - start_stance) / fs <= min_stance_time:
                    statef[start_stance:i] = 1
                start_swing = i
                wmax = 0

            if np.linalg.norm(w[i, :]) > wmax:
                wmax = np.linalg.norm(w[i, :])

            if state[i] - state[i - 1] == -1:  # 1 (swing) -> 0 (stance)
                # deal with false positive
                if wmax < swing_thresh:
                    statef[start_swing:i] = 0
                start_stance = i

            # Estimate and save the yaw of the sensor (different than direction of travel)
            # unused here but potentially useful for orienting a GUI
            heading[i] = arctan2(C[1, 0], C[0, 0])
            C_prev = C.copy()  # Save orientation estimate, required at start of main loop

            # compute horizontal distance
            distance[i] = distance[i - 1] + np.sqrt((pos_n[0, i] - pos_n[0, i - 1]) ** 2 +
                                                    (pos_n[1, i] - pos_n[0, i - 1]) ** 2)

        return acc_n, vel_n, pos_n, state, statef



    def process_data(self):
        """
        Process the raw data and run it through a Kalman Filter to obtain heading and other signals
        used to calculate gait parameters
        """
        # sensors used
        sensors = ['dorsal_foot_right', 'dorsal_foot_left']

        for s in ['1', '2']:  # self.subs:
            for l in sensors:
                for e in self.events:
                    if np.linalg.norm(self.g[s][l]) < 1.5:  # if the magnitude of gravity is less than 1.5
                        # ie gravity units are in G's
                        self.raw_data[s][l]['accel'][e][:, 1:] *= 9.81  # convert to m/s^2 from g
                        self.g[s][l] *= 9.81  # convert to m/s^2 from g
                    if self.raw_data[s][l]['gyro'][e][:, 1:].max() > 50:  # if data is clearly in deg/s
                        self.raw_data[s][l]['gyro'][e][:, 1:] *= np.pi/180  # convert to rad/s

                    zed, zind = GaitParameters._calculate_stillness_factor(self.raw_data[s][l]['accel'][e][:, 1:],
                                                                           self.raw_data[s][l]['gyro'][e][:, 1:],
                                                                           self.g[s][l], self.zt)

                    a_n, v_n, p_n, state, statef = GaitParameters._kalman_filter(self.raw_data[s][l]['accel'][e][:, 0],
                                                                                 self.raw_data[s][l]['accel'][e][:, 1:],
                                                                                 self.raw_data[s][l]['gyro'][e][:, 1:],
                                                                                 self.g[s][l], zind, self.mst, self.swt)

raw_data = MC10py.OpenMC10('C:\\Users\\Lukas Adamowicz\\Documents\\Study Data\\EMT\\ASYM_OFFICIAL\\data.pickle')
test = GaitParameters(raw_data, event_match='Walk and Turn', alt_still='Blind Standing')
test._calibration_detect(still_time=6)
test._turn_detect(plot=True)
# test.process_data()
