
import datetime
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from PyDSS.utils.utils import load_data, dump_data
from PyDSS.pydss_project import PyDssProject
from PyDSS.pydss_results import PyDssResults
from tests.common import CUSTOM_EXPORTS_PROJECT_PATH, cleanup_project, \
    run_project_with_custom_exports
from PyDSS.common import SIMULATION_SETTINGS_FILENAME


def test_custom_exports(cleanup_project):
    PyDssProject.run_project(
        CUSTOM_EXPORTS_PROJECT_PATH,
        simulation_file=SIMULATION_SETTINGS_FILENAME,
    )
    results = PyDssResults(CUSTOM_EXPORTS_PROJECT_PATH)
    assert len(results.scenarios) == 1
    scenario = results.scenarios[0]

    # Property stored at all time points.
    df = scenario.get_full_dataframe("Buses", "puVmagAngle")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 96

    # Property stored with a moving average.
    df = scenario.get_dataframe("Buses", "DistanceAvg", "t9")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == int(96)
    #assert len(df) == int(96 / 5)
    for val in df.iloc[9:, 0]:
        assert round(val, 3) == 0.082

    # Filtered value on custom function.
    df = scenario.get_dataframe("Lines", "LoadingPercent", "Line.sl_22")
    assert len(df) == 17

    df = scenario.get_dataframe("Lines", "LoadingPercentAvg", "Line.sl_22")
    # This was computed from raw data.
    assert len(df) == 15
    # TODO incorrect after more decimal points
    assert round(df.iloc[:, 0].values[14], 3) == 21.803

    # Subset of names. VoltagesMagAng has specific names, CurrentsMagAng has regex
    for name in ("Line.pvl_110", "Line.pvl_111", "Line.pvl_112", "Line.pvl_113"):
        properties = scenario.list_element_properties("Lines", element_name=name)
        assert "VoltagesMagAng" in properties
        assert "CurrentsMagAng" in properties

    properties  = scenario.list_element_properties("Lines", element_name="Line.SL_14")
    assert "VoltagesMagAng" not in properties
    assert "CurrentsMagAng" not in properties

    ## Two types of sums are stored.
    normal_amps_sum = scenario.get_element_property_number("Lines", "NormalAmpsSum", "Line.pvl_110")
    assert normal_amps_sum == 96 * 65.0
    scenario.get_element_property_number("Lines", "CurrentsSum", "Line.pvl_110")
    scenario.get_element_property_number("Circuits", "LossesSum", "Circuit.heco19021")

    sums_json = os.path.join(
        CUSTOM_EXPORTS_PROJECT_PATH,
        "Exports",
        "scenario1",
        "element_property_numbers.json"
    )
    assert os.path.exists(sums_json)
    data = load_data(sums_json)
    assert data

    pv_profiles = scenario.read_pv_profiles()
    assert pv_profiles["pv_systems"]
    for info in pv_profiles["pv_systems"]:
        assert isinstance(info["name"], str)
        assert isinstance(info["irradiance"], float)
        assert isinstance(info["pmpp"], float)
        assert isinstance(info["load_shape_profile"], str)
        assert isinstance(info["load_shape_pmult_sum"], float)


def test_export_moving_averages(cleanup_project):
    # Compares the moving average storage/calculation with a rolling average
    # computed on dataset with every time point.
    path = CUSTOM_EXPORTS_PROJECT_PATH
    sim_file = SIMULATION_SETTINGS_FILENAME
    circuit = "Circuit.heco19021"
    window_size = 10
    PyDssProject.run_project(path, simulation_file=sim_file)

    # This DataFrame will have values at every time point.
    df1 = _get_dataframe(path, "Circuits", "LineLosses", circuit)
    assert len(df1) == 96
    df1_rm = df1.rolling(window_size).mean()

    data = {
        "Circuits": {
            "LineLosses": {
                "store_values_type": "moving_average",
                "window_size": window_size,
            },
        }
    }
    run_project_with_custom_exports(path, "scenario1", sim_file, data)
    results = PyDssResults(path)
    assert len(results.scenarios) == 1
    scenario = results.scenarios[0]

    # This DataFrame will have moving averages.
    df2 = _get_dataframe(path, "Circuits", "LineLossesAvg", circuit)
    assert len(df2) == 96

    for val1, val2 in zip(df1_rm.iloc[:, 0].values, df2.iloc[:, 0].values):
        if np.isnan(val1):
            assert np.isnan(val2)
        else:
            assert round(val2, 5) == round(val1, 5)


def _get_dataframe(path, elem_class, prop, name):
    results = PyDssResults(path)
    assert len(results.scenarios) == 1
    scenario = results.scenarios[0]
    return scenario.get_dataframe(elem_class, prop, name, real_only=True)
