
from collections import deque
import logging
import math

import numpy as np
import pandas as pd

from PyDSS.exceptions import InvalidParameter
from PyDSS.value_storage import ValueContainer, ValueByNumber


logger = logging.getLogger(__name__)


def calculate_line_loading_percent(line, timestamp, step_number, options):
    normal_amps = line.GetValue("NormalAmps", convert=True).value
    currents = line.GetValue("Currents", convert=True).value
    current = max([math.sqrt(x.real**2 + x.imag**2) for x in currents])
    loading = current / normal_amps * 100
    return ValueByNumber(line.Name, "LineLoading", loading)


def calculate_transformer_loading_percent(
        transformer, timestamp, step_number, options
    ):
    normal_amps = transformer.GetValue("NormalAmps", convert=True).value
    currents = transformer.GetValue("Currents", convert=True).value
    current = max([math.sqrt(x.real**2 + x.imag**2) for x in currents])
    loading = current / normal_amps * 100
    return ValueByNumber(transformer.Name, "TransformerLoading", loading)


def track_capacitor_state_changes(
        capacitor, timestamp, step_number, options, last_value, count
    ):
    capacitor.dss.Capacitors.Name(capacitor.Name)
    if capacitor.dss.CktElement.Name() != capacitor.dss.Element.Name():
        raise InvalidParameter(
            f"Object is not a circuit element {capacitor.Name}"
        )
    states = capacitor._dssInstance.Capacitors.States()
    if states == -1:
        raise Exception(
            f"failed to get Capacitors.States() for {capacitor.Name}"
        )

    cur_value = sum(states)
    if last_value is None and cur_value != last_value:
        logger.debug("%s changed state old=%s new=%s", capacitor.Name,
                     last_value, cur_value)
        count += 1

    return cur_value, count


def track_reg_control_tap_number_changes(
        reg_control, timestamp, step_number, options, last_value, count
    ):
    reg_control.dss.RegControls.Name(reg_control.Name)
    if reg_control.dss.CktElement.Name() != reg_control.dss.Element.Name():
        raise InvalidParameter(
            f"Object is not a circuit element {reg_control.Name()}"
        )
    tap_number = reg_control.dss.RegControls.TapNumber()
    if last_value is not None:
        count += abs(tap_number - last_value)
    logger.debug("%s changed count from %s to %s count=%s", reg_control.Name,
                 last_value, tap_number, count)
    return tap_number, count


class CircularBufferHelper:
    def __init__(self, prop):
        self._buf = deque(maxlen=prop.window_size)
        self._window_size = prop.window_size

    def __len__(self):
        return len(self._buf)

    def append(self, val):
        self._buf.append(val)

    def average(self):
        assert self._buf
        if isinstance(self._buf[0], list):
            return pd.DataFrame(self._buf).rolling(self._window_size).mean().values
        if len(self._buf) < self._window_size:
            return np.NaN
        return sum(self._buf) / len(self._buf)
