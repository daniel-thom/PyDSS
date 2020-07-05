
import datetime
import os
import re
import shutil
import tempfile

import pandas as pd

from PyDSS.utils.dataframe_utils import read_dataframe
from PyDSS.utils.utils import load_data, dump_data
from PyDSS.pydss_project import PyDssProject
from PyDSS.reports import ReportGranularity
from PyDSS.pydss_results import PyDssResults
from tests.common import PV_REPORTS_PROJECT_PATH, cleanup_project
from PyDSS.common import SIMULATION_SETTINGS_FILENAME


BASE_FILENAME = os.path.join(PV_REPORTS_PROJECT_PATH, SIMULATION_SETTINGS_FILENAME)
TEST_SIM_BASE_NAME = "test_sim.toml"
TEST_FILENAME = os.path.join(PV_REPORTS_PROJECT_PATH, TEST_SIM_BASE_NAME)


def test_pv_reports_per_element_per_time_point(cleanup_project):
    granularities = [x for x in ReportGranularity]
    for granularity in granularities:
        data = load_data(BASE_FILENAME)
        data["Reports"]["Granularity"] = granularity.value
        dump_data(data, TEST_FILENAME)
        try:
            PyDssProject.run_project(
                PV_REPORTS_PROJECT_PATH,
                simulation_file=TEST_SIM_BASE_NAME,
            )
            verify_reports(granularity)
        finally:
            os.remove(TEST_FILENAME)


def verify_reports(granularity):
    results = PyDssResults(PV_REPORTS_PROJECT_PATH)

    # This test data doesn't have changes for Capacitors or RegControls.
    capacitor_change_counts = results.read_report("Capacitor State Change Counts")
    assert len(capacitor_change_counts["scenarios"]) == 2
    assert not capacitor_change_counts["scenarios"][1]["capacitors"]

    reg_control_change_counts = results.read_report("RegControl Tap Number Change Counts")
    assert len(reg_control_change_counts["scenarios"]) == 2
    assert not reg_control_change_counts["scenarios"][1]["reg_controls"]

    if granularity in (
        ReportGranularity.PER_ELEMENT_PER_TIME_POINT,
        ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT,
    ):
        clipping_name = os.path.join(PV_REPORTS_PROJECT_PATH, "Reports", "pv_clipping.h5")
        clipping = read_dataframe(clipping_name)
        curtailment_name = os.path.join(PV_REPORTS_PROJECT_PATH, "Reports", "pv_curtailment.h5")
        curtailment = read_dataframe(curtailment_name)
    else:
        clipping_name = os.path.join(PV_REPORTS_PROJECT_PATH, "Reports", "pv_clipping.json")
        clipping = load_data(clipping_name)
        curtailment_name = os.path.join(PV_REPORTS_PROJECT_PATH, "Reports", "pv_curtailment.json")
        curtailment = load_data(curtailment_name)

    # TODO: thermal metrics, voltage metrics
