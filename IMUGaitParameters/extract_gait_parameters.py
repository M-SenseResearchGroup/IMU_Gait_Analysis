from matplotlib import pyplot as pl
from scipy import signal
import numpy as np
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
            self.alt_events = [e for e in self.raw_data[self.subs[0]][self.sens[0]]['accel'].keys()
                               if self.astill in e]
        else:
            self.alt_events = None

        # PRE-ALLOCATE storage for future use
        self.g = {i: {j: None for j in self.raw_data[i].keys()} for i in self.subs}  # standing gravity vector
        # self.data = {i: dict() for i in self.subs}  # pre-allocate storage for data
        self.gait_params = {i: dict() for i in self.subs}  # pre-allocate storage for gait parameters

    def _mean_stillness(self, bias, i, ax, s, sensor, e, nst, plot):
        mag = np.linalg.norm(self.raw_data[s][sensor]['gyro'][e][:, 1:], axis=1)  # ang. vel. magnitude

        # calculate moving mean
        _mm = np.cumsum(mag)
        mm = _mm[nst - 1:]
        mm[1:] = _mm[nst:] - _mm[:-nst]
        mm /= nst

        # determine minimum moving mean location
        mind = np.argmin(mm) + nst - 1
        var = np.var(mag[mind - nst:mind])

        # store bias
        for l in self.raw_data[s].keys():
            bias[l][i, :] = np.mean(self.raw_data[s][l]['gyro'][e][mind - nst:mind, 1:], axis=0)

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
            fr = 1000/(np.mean(np.diff(self.raw_data[s][sensor]['gyro'][self.events[0]][:, 0])))
            fnyq = fr/2  # nyquist frequency
            n1 = int(round(fr))  # samples in 1 second
            nst = int(round(still_time * n1))  # samples in still_time seconds

            i = 0

            if self.alt_events is not None:
                # allocate storage for bias in each sensor
                bias = {l: np.zeros((len(self.alt_events), 3)) for l in self.raw_data[s].keys()}

                if plot:
                    f, ax = pl.subplots(len(self.alt_events), figsize=(14, 8))
                    f.suptitle(f'Subject: {s}')
                else:
                    ax = None
                for e in self.alt_events:
                    i, bias = self._mean_stillness(bias, i, ax, s, sensor, e, nst, plot)

                if plot:
                    f.tight_layout(rect=[0, 0.03, 1, 0.95])
            else:
                # allocate storage for bias in each sensor
                bias = {l: np.zeros((self.n_ev, 3)) for l in self.raw_data[s].keys()}

                if plot:
                    f, ax = pl.subplots(self.n_ev, figsize=(14, 8))
                    f.suptitle(f'Subject: {s}')
                else:
                    ax = None
                for e in self.events:
                    i, bias = self._mean_stillness(bias, i, ax, s, sensor, e, nst, plot)

                if plot:
                    f.tight_layout(rect=[0, 0.03, 1, 0.95])

            # remove bias from all sensors and events
            for l in self.raw_data[s].keys():
                for e in self.raw_data[s][l]['gyro'].keys():
                    self.raw_data[s][l]['gyro'][e][:, 1:] -= np.mean(bias[l], axis=0)

    def _determine_gravity(self, plot=False):
        """
        Determine gravity for trials of interest

        Parameters
        ----------
        plot : bool, optional
            Plot results.  Defaults to False.
        """
        b, a = signal.ellip(5, 1, 60, 1/31.25)

        for s in ['1', '2']:
            if plot:
                f, ax = pl.subplots(self.n_ev, figsize=(13, 8))

    def _turn_detect(self, plot=False):
        pl.close('all')  # close all open plots before running

        for s in self.subs:  # iterate through each subject
            if plot:
                f, ax = pl.subplots(self.n_ev, figsize=(14, 9))

            i = 0
            for e in self.events:
                pass  # do after finding still spot in data?


raw_data = MC10py.OpenMC10('C:\\Users\\Lukas Adamowicz\\Documents\\Study Data\\EMT\\ASYM_OFFICIAL\\data.pickle')
test = GaitParameters(raw_data, event_match='Walk and Turn', alt_still='Blind Standing')
test._calibration_detect(still_time=6)
# test._determine_gravity(plot=True)
