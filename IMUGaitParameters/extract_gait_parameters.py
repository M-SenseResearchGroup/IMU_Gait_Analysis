class GaitParameters:
    """
    Class for storing methods to calculate gait parameters from IMU data.  Minimum necessary IMUs are on both feet and
    the sacrum
    """
    def __init__(self, data, z_thresh=1.25, min_stance_time=0.03, swing_thresh=2):
        """
        Parameters
        ----------
        data : dict
            Dictionary of data for all subjects.
        z_thresh : float, optional
            Threshold for z values.  Default is 1.25
        min_stance_time : float, optional
            Minimum stance time in seconds for detection.  Defaults to 0.03 seconds.
        swing_thresh : float, int, optional
            Threshold for swing detection
        """
        # assign variables to class attributes
        self.data = data
        self.zt = z_thresh
        self.mst = min_stance_time
        self.swt = swing_thresh

