from matplotlib import pyplot as pl
from scipy.signal import butter, filtfilt
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
        self.data = data
        self.zt = z_thresh
        self.mst = min_stance_time
        self.swt = swing_thresh
        self.ev_match = event_match

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
        for s in _subs:
            # assume all sensors recorded the same events
            _data_check.append(len([e for e in self.data[s][self.sens[0]]['accel'].keys() if self.ev_match in e]))

        self.n_ev, _ = stats.mode(_data_check)
        for i in range(len(_data_check)):
            if _data_check[i] < self.n_ev:
                print(f'Removing subject {_subs[i]} data due to missing events.')
                self.data.pop(_subs[i])

        self.subs = list(self.data.keys())  # get final list of subjects

        self.gait_params = {i: dict() for i in self.data.keys()}  # pre-allocate storage for gait parameters

    def _turn_detect(self, plot=False):
        pl.close('all')  # close all open plots before running

        # for s in self.data.keys():  # iterate through each subject
        #     nev = sum(self.data[s][])
        #     if plot:
        #         f, ax = pl.subplots(4, figsize=(14, 9))
        #
        #     i = 0


raw_data = MC10py.OpenMC10('C:\\Users\\Lukas Adamowicz\\Documents\\Study Data\\EMT\\ASYM_OFFICIAL\\data.pickle')
GaitParameters(raw_data, event_match='Walk and Turn')
