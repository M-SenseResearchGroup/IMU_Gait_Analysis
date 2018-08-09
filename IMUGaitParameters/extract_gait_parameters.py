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

    def __init__(self, data, z_thresh=1.25, min_stance_time=0.03, swing_thresh=2, event_match='', turn_split=False,
                 turn_kwargs={}):
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
        for s in _subs:
            # assume all sensors recorded the same events
            _data_check.append(len([e for e in self.raw_data[s][self.sens[0]]['accel'].keys() if event_match in e]))

        self.n_ev = stats.mode(_data_check)[0][0]
        for i in range(len(_data_check)):
            if _data_check[i] < self.n_ev:
                print(f'Removing subject {_subs[i]} data due to missing events.')
                self.raw_data.pop(_subs[i])

        self.subs = list(self.raw_data.keys())  # get final list of subjects
        self.events = [e for e in self.raw_data[self.subs[0]][self.sens[0]]['accel'].keys() if event_match in e]

        # PRE-ALLOCATE storage for future use
        self.data = {i: dict() for i in self.subs}  # pre-allocate storage for data
        self.gait_params = {i: dict() for i in self.subs}  # pre-allocate storage for gait parameters

    # TODO add kwargs for still period search time, and window length
    def _calibration_detect(self):
        """
        Detect the calibration point in the data.  Should be a still point at the beginning of the trial
        """
        pl.close('all')
        for s in ['1', '2']:
            fr = 1000/(np.mean(np.diff(self.raw_data[s]['sacrum']['gyro'][self.events[0]][:, 0])))
            nfr = int(round(fr))
            b, a = signal.butter(4, 4/fr)
            f, ax = pl.subplots(self.n_ev, figsize=(16, 9))
            i = 0
            for e in self.events:
                # filter the first 2 seconds of acceleration magnitude
                maf = signal.filtfilt(b, a, np.sqrt(np.sum(self.raw_data[s]['sacrum']['accel'][e][:2*nfr+1, 1:]**2,
                                                           axis=1)))
                rmse = np.sqrt((maf-1.0)**2)  # RMS error of first 2 seconds of accel data
                cs_rmse = np.cumsum(rmse, dtype=float)  # cumulative sum of RMSE

                nhalf = int(round(nfr/2))  # number of samples in 1/2 second
                sum_rmse = np.insert(cs_rmse[nhalf:] - cs_rmse[:-nhalf], 0, cs_rmse[nhalf-1])  # sum over 0.5s windows

                min_rmse_ind = np.argmin(sum_rmse)  # get the index of the window with the minimum RMSE

                ax[i].plot(self.raw_data[s]['sacrum']['accel'][e][:2*nfr+1, 0],
                           np.sqrt(np.sum(self.raw_data[s]['sacrum']['accel'][e][:2*nfr+1, 1:]**2, axis=1)),
                           label='raw data')

                ax[i].plot(self.raw_data[s]['sacrum']['accel'][e][:2*nfr+1, 0], maf, label='filtered')
                ax[i].plot(self.raw_data[s]['sacrum']['accel'][e][min_rmse_ind:min_rmse_ind+nhalf, 0],
                           maf[min_rmse_ind:min_rmse_ind+nhalf], 'r', alpha=0.4, linewidth=6,
                           label="""standing 'still'""")

                ax[i].set_ylabel('Accel (g)')
                ax[i].legend(loc='best')

                i += 1

    def _turn_detect(self, plot=False):
        pl.close('all')  # close all open plots before running

        for s in self.subs:  # iterate through each subject
            if plot:
                f, ax = pl.subplots(self.n_ev, figsize=(14, 9))

            i = 0
            for e in self.events:
                pass  # do after finding still spot in data?


raw_data = MC10py.OpenMC10('C:\\Users\\Lukas Adamowicz\\Documents\\Study Data\\EMT\\ASYM_OFFICIAL\\data.pickle')
test = GaitParameters(raw_data, event_match='Walk and Turn')
test._calibration_detect()
