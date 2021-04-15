
import logging
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import opendssdirect as dss
import pandas as pd

from PyDSS.common import DATE_FORMAT, TIME_FORMAT


logger = logging.getLogger(__name__)


class CircularBufferHelper:
    def __init__(self, window_size):
        self._buf = deque(maxlen=window_size)
        self._window_size = window_size

    def __len__(self):
        return len(self._buf)

    def append(self, val):
        self._buf.append(val)

    def average(self):
        if len(self._buf) < self._window_size:
            return np.NaN
        return sum(self._buf) / len(self._buf)


class SimulationFilteredTimeRange:
    """Provides filtering in a time range."""
    def __init__(self, start, end):
        self._start = start
        self._end = end
        default_start = time.strptime("00:00:00", TIME_FORMAT)
        default_end = time.strptime("11:59:59", TIME_FORMAT)
        self._no_filtering = start == default_start and end == default_end

    @classmethod
    def from_settings(cls, settings):
        """Return SimulationFilteredRange from the simulation settings.

        Parameters
        ----------
        settings : dict
            settings from project simulation.toml

        Returns
        -------
        SimulationFilteredRange

        """
        start = time.strptime(settings["Project"]["Simulation range"]["start"], TIME_FORMAT)
        end = time.strptime(settings["Project"]["Simulation range"]["end"], TIME_FORMAT)
        return cls(start=start, end=end)

    def is_within_range(self, timestamp):
        """Return True if the timestamp is within the filtered range.

        Parameters
        ----------
        timestamp : datetime

        Returns
        -------
        bool

        """
        if self._no_filtering:
            return True

        ts = time.struct_time((
            self._start.tm_year, self._start.tm_mon, self._start.tm_mday, timestamp.hour,
            timestamp.minute, timestamp.second, self._start.tm_wday, self._start.tm_yday,
            self._start.tm_isdst
        ))
        return ts >= self._start and ts <=self._end


def get_start_time(settings):
    """Return the start time of the simulation.

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    datetime

    """
    return datetime.strptime(settings['Project']["Start time"], DATE_FORMAT)


def get_simulation_resolution(settings):
    """Return the simulation of the resolution

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    datetime

    """
    return timedelta(seconds=settings["Project"]["Step resolution (sec)"])


def create_time_range_from_settings(settings):
    """Return the start time, step time, and end time from the settings.

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    tuple
        (start, end, step)

    """
    start_time = get_start_time(settings)
    end_time = start_time + timedelta(minutes=settings["Project"]["Simulation duration (min)"])
    step_time = get_simulation_resolution(settings)
    return start_time, end_time, step_time


def create_datetime_index_from_settings(settings):
    """Return time indices created from the simulation settings.

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    pd.DatetimeIndex

    """
    start_time, end_time, step_time = create_time_range_from_settings(settings)
    data = []
    cur_time = start_time
    while cur_time < end_time:
        data.append(cur_time)
        cur_time += step_time

    return pd.DatetimeIndex(data)


def create_loadshape_pmult_dataframe(settings):
    """Return a loadshape dataframe representing all available data.
    This assumes that a loadshape has been selected in OpenDSS.

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    pd.DatetimeIndex

    """
    start_time = datetime.strptime(settings['Project']['Loadshape start time'], DATE_FORMAT)
    data = dss.LoadShape.PMult()
    interval = timedelta(seconds=dss.LoadShape.SInterval())
    npts = dss.LoadShape.Npts()

    indices = []
    cur_time = start_time
    for _ in range(npts):
        indices.append(cur_time)
        cur_time += interval

    return pd.DataFrame(data, index=pd.DatetimeIndex(indices))


def create_loadshape_pmult_dataframe_for_simulation(settings):
    """Return a loadshape pmult dataframe that only contains time points used
    by the simulation.
    This assumes that a loadshape has been selected in OpenDSS.

    Parameters
    ----------
    settings : dict
        settings from project simulation.toml

    Returns
    -------
    pd.DataFrame

    """
    df = create_loadshape_pmult_dataframe(settings)
    simulation_index = create_datetime_index_from_settings(settings)
    return df.loc[simulation_index]
