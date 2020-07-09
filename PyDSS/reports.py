"""Creates reports on data exported by PyDSS"""

import abc
import copy
from datetime import timedelta
import enum
import logging
import os

import pandas as pd

from PyDSS.common import PV_LOAD_SHAPE_FILENAME
from PyDSS.exceptions import InvalidConfiguration, InvalidParameter
from PyDSS.utils.dataframe_utils import read_dataframe, write_dataframe
from PyDSS.utils.utils import dump_data


REPORTS_DIR = "Reports"

logger = logging.getLogger(__name__)


class Reports:
    """Generate reports from a PyDSS project"""
    def __init__(self, results):
        self._results = results
        self._report_names = []
        self._simulation_config = results.simulation_config
        for report in results.simulation_config["Reports"]["Types"]:
            if report["enabled"]:
                self._report_names.append(report["name"])
        self._output_dir = os.path.join(results.project_path, REPORTS_DIR)
        os.makedirs(self._output_dir, exist_ok=True)

    @staticmethod
    def append_required_exports(exports, options):
        """Append export properties required by the configured reports.

        Parameters
        ----------
        exports : ExportListReader
        options : dict
            Simulation options

        """
        report_options = options.get("Reports")
        if report_options is None:
            return

        for report in report_options["Types"]:
            if not report["enabled"]:
                continue
            name = report["name"]
            if name not in REPORTS:
                raise InvalidConfiguration(f"{name} is not a valid report")

            scenarios = report.get("scenarios")
            active_scenario = options["Project"]["Active Scenario"]
            if scenarios and active_scenario not in scenarios:
                logger.debug("report %s is not enabled for scenario %s", name,
                             active_scenario)
                continue

            required = REPORTS[name].get_required_exports(options)
            for elem_class, required_properties in required.items():
                for req_prop in required_properties:
                    found = False
                    store_type = req_prop["store_values_type"]
                    for prop in exports.list_element_properties(elem_class):
                        if prop.name == req_prop["property"] and \
                                prop.store_values_type.value == store_type:
                            found = True
                            break
                    if not found:
                        exports.append_property(elem_class, req_prop)
                        logger.debug("Add required property: %s %s", elem_class, req_prop)


    @classmethod
    def generate_reports(cls, results):
        """Generate all reports specified in the configuration.

        Parameters
        ----------
        results : PyDssResults

        Returns
        -------
        list
            list of report filenames

        """
        reports = Reports(results)
        return reports.generate()

    def generate(self):
        """Generate all reports specified in the configuration.

        Returns
        -------
        list
            list of report filenames

        """
        filenames = []
        for name in self._report_names:
            report = REPORTS[name](name, self._results, self._simulation_config)
            filename = report.generate(self._output_dir)
            filenames.append(filename)

        return filenames


class ReportBase(abc.ABC):
    """Base class for reports"""
    def __init__(self, name, results, simulation_config):
        self._results = results
        self._simulation_config = simulation_config
        self._report_global_options = simulation_config["Reports"]
        self._report_options = None
        self._report_options = _get_report_options(simulation_config, name)

    @abc.abstractmethod
    def generate(self, output_dir):
        """Generate a report in output_dir.

        Returns
        -------
        str
            path to report

        """

    @staticmethod
    @abc.abstractmethod
    def get_required_exports(simulation_config):
        """Return the properties required for the report for export.

        Parameters
        ----------
        simulation_config: dict
            settings from simulation config file

        Returns
        -------
        dict

        """

    @staticmethod
    def _params_from_granularity(granularity):
        if granularity == ReportGranularity.PER_ELEMENT_PER_TIME_POINT:
            store_values_type = "all"
            sum_elements = False
        elif granularity == ReportGranularity.PER_ELEMENT_TOTAL:
            store_values_type = "sum"
            sum_elements = False
        elif granularity == ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT:
            store_values_type = "all"
            sum_elements = True
        elif granularity == ReportGranularity.ALL_ELEMENTS_TOTAL:
            store_values_type = "sum"
            sum_elements = True
        else:
            assert False, str(granularity)

        return store_values_type, sum_elements


class CapacitorStateChangeReport(ReportBase):
    """Reports the state changes per Capacitor."""

    FILENAME = "capacitor_state_changes.json"
    NAME = "Capacitor State Change Counts"

    def generate(self, output_dir):
        data = {"scenarios": []}
        for scenario in self._results.scenarios:
            scenario_data = {"name": scenario.name, "capacitors": []}
            for capacitor in scenario.list_element_names("Capacitors"):
                try:
                    change_count = int(scenario.get_element_property_number(
                        "Capacitors", "TrackStateChanges", capacitor
                    ))
                except InvalidParameter:
                    change_count = 0
                changes = {"name": capacitor, "change_count": change_count}
                scenario_data["capacitors"].append(changes)
            data["scenarios"].append(scenario_data)

        filename = os.path.join(output_dir, self.FILENAME)
        dump_data(data, filename, indent=2)
        logger.info("Generated %s", filename)
        return filename

    @staticmethod
    def get_required_exports(simulation_config):
        return {
            "Capacitors": [
                {
                    "property": "TrackStateChanges",
                    "store_values_type": "change_count",
                }
            ]
        }


class FeederLossesReport(ReportBase):
    """Reports the feeder losses."""

    FILENAME = "feeder_losses.json"
    NAME = "Feeder Losses"

    def generate(self, output_dir):
        assert len(self._results.scenarios) == 2
        scenario = self._results.scenarios[1]
        total_losses_dict = scenario.get_summed_element_total("Circuits", "LossesSum")
        total_losses = abs(next(iter(total_losses_dict.values())))
        line_losses_dict = scenario.get_summed_element_total("Circuits", "LineLossesSum")
        line_losses = abs(next(iter(line_losses_dict.values())))
        transformer_losses = total_losses - line_losses
        total_load_power_dict = scenario.get_summed_element_total("Loads", "PowersSum")
        total_load_power = 0
        for val in total_load_power_dict.values():
            total_load_power += val.real

        data = {
            "total_losses": total_losses,
            "line_losses": line_losses,
            "tranformer_losses": transformer_losses,
            "total_load_demand": total_load_power,
            # TODO: total losses as a percentage of total load demand?
        }
        filename = os.path.join(output_dir, self.FILENAME)
        dump_data(data, filename, indent=2)
        logger.info("Generated %s", filename)
        return filename

    @staticmethod
    def get_required_exports(simulation_config):
        return {
            "Circuits": [
                {
                    "property": "Losses",
                    "store_values_type": "sum",
                    "sum_elements": True,
                },
                {
                    "property": "LineLosses",
                    "store_values_type": "sum",
                    "sum_elements": True,
                }
            ],
            "Loads": [
                {
                    "property": "Powers",
                    "store_values_type": "sum",
                    "sum_elements": True,
                    "data_conversion": "abs_sum",
                },
            ]
        }


class PvReportBase(ReportBase, abc.ABC):
    """Base class for PV reports"""
    def __init__(self, name, results, simulation_config):
        super().__init__(name, results, simulation_config)
        assert len(results.scenarios) == 2
        self._pf1_scenario = results.scenarios[0]
        self._control_mode_scenario = results.scenarios[1]
        self._pv_system_names = self._control_mode_scenario.list_element_names("PVSystems")
        self._pf1_pv_systems = {
            x["name"]: x for x in self._pf1_scenario.read_pv_profiles()["pv_systems"]
        }
        self._control_mode_pv_systems = {
            x["name"]: x for x in self._control_mode_scenario.read_pv_profiles()["pv_systems"]
        }

    def _get_pv_system_info(self, pv_system, scenario):
        if scenario == "pf1":
            pv_systems = self._pf1_pv_systems
        else:
            pv_systems = self._control_mode_pv_systems

        return pv_systems[pv_system]

    @staticmethod
    def get_required_exports(simulation_config):
        granularity = ReportGranularity(simulation_config["Reports"]["Granularity"])
        _type, sum_elements = ReportBase._params_from_granularity(granularity)
        return {
            "PVSystems": [
                {
                    "property": "Powers",
                    "data_conversion": "abs_sum",
                    "store_values_type": _type,
                    "sum_elements": sum_elements,
                },
                {
                    "property": "ExportPowersMetric",
                    "store_values_type": _type,
                    "sum_elements": sum_elements,
                }
            ]
        }


class PvClippingReport(PvReportBase):
    """Reports PV Clipping for the simulation."""

    PER_TIME_POINT_FILENAME = "pv_clipping.h5"
    TOTAL_FILENAME = "pv_clipping.json"
    NAME = "PV Clipping"

    def _generate_per_pv_system_per_time_point(self, pv_load_shapes, output_dir):
        data = {}
        index = None
        for name in self._pv_system_names:
            cm_info = self._get_pv_system_info(name, "control_mode")
            dc_power_per_time_point = pv_load_shapes[cm_info["load_shape_profile"]]
            pf1_real_power = self._pf1_scenario.get_dataframe("PVSystems", "Powers", name)
            if index is None:
                index = pf1_real_power.index
            if len(dc_power_per_time_point) > len(pf1_real_power):
                dc_power_per_time_point = dc_power_per_time_point[:len(pf1_real_power)]
            clipping = dc_power_per_time_point - pf1_real_power.values[0]
            data[name] = clipping.values

        # TODO: datetime is off by 6 hours 15 minutes
        df = pd.DataFrame(data, index=index)
        filename = os.path.join(output_dir, self.PER_TIME_POINT_FILENAME)
        write_dataframe(df, filename, compress=True)

    def _generate_per_pv_system_total(self, output_dir):
        data = {"pv_systems": []}
        for name in self._pv_system_names:
            cm_info = self._get_pv_system_info(name, "control_mode")
            pmpp = cm_info["pmpp"]
            irradiance = cm_info["irradiance"]
            total_irradiance = cm_info["load_shape_pmult_sum"]
            annual_dc_power = pmpp * irradiance * total_irradiance
            pf1_real_power = self._pf1_scenario.get_element_property_number(
                "PVSystems", "PowersSum", name
            )
            clipping = annual_dc_power - pf1_real_power
            data["pv_systems"].append(
                {
                    "name": name,
                    "clipping": clipping,
                }
            )

        filename = os.path.join(output_dir, self.TOTAL_FILENAME)
        dump_data(data, filename, indent=2)

    def _generate_all_pv_systems_per_time_point(self, pv_load_shapes, output_dir):
        pf1_real_power = self._pf1_scenario.get_summed_element_dataframe(
            "PVSystems", "Powers"
        )
        # TODO: do we need to verify that there are no extra pv systems?
        dc_power_per_time_point = []
        for _, row in pv_load_shapes.iterrows():
            dc_power_per_time_point.append(row.sum())
            if len(dc_power_per_time_point) == len(pf1_real_power):
                break
        clipping = pd.DataFrame(
            dc_power_per_time_point - pf1_real_power.values[0],
            index=pf1_real_power.index,
            columns=["Clipping"],
        )
        filename = os.path.join(output_dir, self.PER_TIME_POINT_FILENAME)
        write_dataframe(clipping, filename, compress=True)

    def _generate_all_pv_systems_total(self, output_dir):
        annual_dc_power = 0.0
        for name in self._pv_system_names:
            cm_info = self._get_pv_system_info(name, "control_mode")
            pmpp = cm_info["pmpp"]
            irradiance = cm_info["irradiance"]
            total_irradiance = cm_info["load_shape_pmult_sum"]
            annual_dc_power += pmpp * irradiance * total_irradiance

        ac_power = next(iter(
            self._pf1_scenario.get_summed_element_total("PVSystems", "PowersSum").values()
        ))
        # TODO: abs?
        clipping = annual_dc_power - ac_power
            
        data = {"clipping": clipping}
        filename = os.path.join(output_dir, self.TOTAL_FILENAME)
        dump_data(data, filename, indent=2)

    def _read_pv_load_shapes(self):
        path = os.path.join(
            self._simulation_config["Project"]["Project Path"],
            self._simulation_config["Project"]["Active Project"],
            "Exports",
            "control_mode",
            PV_LOAD_SHAPE_FILENAME,
        )
        return read_dataframe(path)

    def generate(self, output_dir):
        granularity = ReportGranularity(self._simulation_config["Reports"]["Granularity"])
        per_time_point = (
            ReportGranularity.PER_ELEMENT_PER_TIME_POINT,
            ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT,
        )
        if granularity in per_time_point:
            pv_load_shapes = self._read_pv_load_shapes()
        else:
            pv_load_shapes = None

        if granularity == ReportGranularity.PER_ELEMENT_PER_TIME_POINT:
            self._generate_per_pv_system_per_time_point(pv_load_shapes, output_dir)
        elif granularity == ReportGranularity.PER_ELEMENT_TOTAL:
            self._generate_per_pv_system_total(output_dir)
        elif granularity == ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT:
            self._generate_all_pv_systems_per_time_point(pv_load_shapes, output_dir)
        elif granularity == ReportGranularity.ALL_ELEMENTS_TOTAL:
            self._generate_all_pv_systems_total(output_dir)
        else:
            assert False


class PvCurtailmentReport(PvReportBase):
    """Reports PV Curtailment at every time point in the simulation."""

    PER_TIME_POINT_FILENAME = "pv_curtailment.h5"
    TOTAL_FILENAME = "pv_curtailment.json"
    NAME = "PV Curtailment"

    def _generate_per_pv_system_per_time_point(self, output_dir):
        pf1_power = self._pf1_scenario.get_full_dataframe("PVSystems", "Powers")
        control_mode_power = self._control_mode_scenario.get_full_dataframe(
            "PVSystems", "Powers"
        )
        df = (pf1_power - control_mode_power) / pf1_power * 100
        filename = os.path.join(output_dir, self.PER_TIME_POINT_FILENAME)
        write_dataframe(df, filename, compress=True)

    def _generate_per_pv_system_total(self, output_dir):
        data = {"pv_systems": []}
        for name in self._pv_system_names:
            pf1_power = self._pf1_scenario.get_element_property_number(
                "PVSystems", "PowersSum", name
            )
            control_mode_power = self._control_mode_scenario.get_element_property_number(
                "PVSystems", "PowersSum", name
            )
            curtailment = (pf1_power - control_mode_power) / pf1_power * 100
            data["pv_systems"].append(
                {
                    "name": name,
                    "curtailment": curtailment,
                }
            )

        filename = os.path.join(output_dir, self.TOTAL_FILENAME)
        dump_data(data, filename, indent=2)

    def _generate_all_pv_systems_per_time_point(self, output_dir):
        pf1_power = self._pf1_scenario.get_summed_element_dataframe("PVSystems", "Powers")
        control_mode_power = self._control_mode_scenario.get_summed_element_dataframe(
            "PVSystems", "Powers"
        )
        df = (pf1_power - control_mode_power) / pf1_power * 100
        filename = os.path.join(output_dir, self.PER_TIME_POINT_FILENAME)
        write_dataframe(df, filename, compress=True)

    def _generate_all_pv_systems_total(self, output_dir):
        pf1_power = next(iter(
            self._pf1_scenario.get_summed_element_total("PVSystems", "PowersSum").values()
        ))
        control_mode_power = next(iter(
            self._control_mode_scenario.get_summed_element_total("PVSystems", "PowersSum").values()
        ))

        curtailment = (pf1_power - control_mode_power) / pf1_power * 100
        data = {"curtailment": curtailment}
        filename = os.path.join(output_dir, self.TOTAL_FILENAME)
        dump_data(data, filename, indent=2)

    def generate(self, output_dir):
        granularity = ReportGranularity(self._simulation_config["Reports"]["Granularity"])
        #per_time_point = (
        #    ReportGranularity.PER_ELEMENT_PER_TIME_POINT,
        #    ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT,
        #)

        if granularity == ReportGranularity.PER_ELEMENT_PER_TIME_POINT:
            self._generate_per_pv_system_per_time_point(output_dir)
        elif granularity == ReportGranularity.PER_ELEMENT_TOTAL:
            self._generate_per_pv_system_total(output_dir)
        elif granularity == ReportGranularity.ALL_ELEMENTS_PER_TIME_POINT:
            self._generate_all_pv_systems_per_time_point(output_dir)
        elif granularity == ReportGranularity.ALL_ELEMENTS_TOTAL:
            self._generate_all_pv_systems_total(output_dir)
        else:
            assert False

    def calculate_pv_curtailment(self):
        """Calculate PV curtailment for all PV systems.

        Returns
        -------
        pd.DataFrame

        """
        pf1_power = self._pf1_scenario.get_full_dataframe(
            "PVSystems", "Powers", real_only=True
        )
        control_mode_power = self._control_mode_scenario.get_full_dataframe(
            "PVSystems", "Powers", real_only=True
        )
        # TODO: needs work
        return (pf1_power - control_mode_power) / pf1_power * 100


class RegControlTapNumberChangeReport(ReportBase):
    """Reports the tap number changes per RegControl."""

    FILENAME = "reg_control_tap_number_changes.json"
    NAME = "RegControl Tap Number Change Counts"

    def generate(self, output_dir):
        data = {"scenarios": []}
        for scenario in self._results.scenarios:
            scenario_data = {"name": scenario.name, "reg_controls": []}
            for reg_control in scenario.list_element_names("RegControls"):
                change_count = int(scenario.get_element_property_number(
                    "RegControls", "TrackTapNumberChanges", reg_control
                ))
                changes = {"name": reg_control, "change_count": change_count}
                scenario_data["reg_controls"].append(changes)
            data["scenarios"].append(scenario_data)

        filename = os.path.join(output_dir, self.FILENAME)
        dump_data(data, filename, indent=2)
        logger.info("Generated %s", filename)
        return filename

    @staticmethod
    def get_required_exports(simulation_config):
        return {
            "RegControls": [
                {
                    "property": "TrackTapNumberChanges",
                    "store_values_type": "change_count",
                }
            ]
        }


class ThermalMetrics(ReportBase):
    """Reports thermal metrics."""

    DEFAULTS = {
        "transformer_loading_percent_threshold": 150,
        "transformer_window_size_hours": 2,
        "transformer_loading_percent_moving_average_threshold": 120,
        "line_window_size_hours": 1,
        "line_loading_percent_threshold": 120,
        "line_loading_percent_moving_average_threshold": 100,
    }
    FILENAME = "thermal_metrics.json"
    NAME = "Thermal Metrics"

    def generate(self, output_dir):
        pass

    @staticmethod
    def get_required_exports(simulation_config):
        resolution = timedelta(seconds=simulation_config["Project"]["Step resolution (sec)"])

        inputs = _get_inputs_from_defaults(simulation_config, ThermalMetrics.NAME)
        if inputs.get("force_instantaneous", False):
            data = ThermalMetrics._get_required_exports_instantaneous(inputs)
        elif inputs.get("force_moving_average", False):
            data = ThermalMetrics._get_required_exports_moving_average(inputs, resolution)
        elif resolution >= timedelta(minutes=15):
            data = ThermalMetrics._get_required_exports_instantaneous(inputs)
        else:
            data = ThermalMetrics._get_required_exports_moving_average(inputs, resolution)

        return data

    @staticmethod
    def _get_required_exports_instantaneous(inputs):
        return {
            "Lines": [
                {
                    "property": "LoadingPercent",
                    "store_values_type": "all",
                    "limits": [0, inputs["line_loading_percent_threshold"]],
                },
            ],
            "Transformers": [
                {
                    "property": "LoadingPercent",
                    "store_values_type": "all",
                    "limits": [0, inputs["transformer_loading_percent_threshold"]],
                },
            ],
        }

    @staticmethod
    def _get_required_exports_moving_average(inputs, resolution):
        transformer_window_size = timedelta(hours=inputs["transformer_window_size_hours"])
        if transformer_window_size % resolution != timedelta(0):
            raise InvalidConfiguration(
                f"transformer_window_size={transformer_window_size} must be a multiple of {resolution}"
            )
        line_window_size = timedelta(hours=inputs["line_window_size_hours"])
        if line_window_size % resolution != timedelta(0):
            raise InvalidConfiguration(
                f"line_window_size={line_window_size} must be a multiple of {resolution}"
            )

        return {
            "Lines": [
                {
                    "property": "LoadingPercent",
                    "store_values_type": "max",
                },
                {
                    "property": "LoadingPercent",
                    "store_values_type": "moving_average_max",
                    "window_size": int(line_window_size / resolution),
                },
            ],
            "Transformers": [
                {
                    "property": "LoadingPercent",
                    "store_values_type": "max",
                },
                {
                    "property": "LoadingPercent",
                    "store_values_type": "moving_average_max",
                    "window_size": int(transformer_window_size / resolution),
                },
            ],
        }


class VoltageMetrics(ReportBase):
    """Reports voltage metrics."""

    DEFAULTS = {
        "inner_limits": [0.95, 1.05],
        "outer_limits": [0.90, 1.0583],
        "window_size_minutes": 10,
    }
    FILENAME = "voltage_metrics.json"
    NAME = "Voltage Metrics"

    def __init__(self, name, results, simulation_config):
        super().__init__(name, results, simulation_config)
        assert len(results.scenarios) == 2
        self._scenario = results.scenarios[1]

    def generate(self, output_dir):
        dfs = self._scenario.get_filtered_dataframes("Nodes", "VoltageMetrics")
        self._gen_metric_1(dfs)

    def _gen_metric_1(self, dfs):
        pass

    @staticmethod
    def get_required_exports(simulation_config):
        resolution = timedelta(seconds=simulation_config["Project"]["Step resolution (sec)"])
        inputs = _get_inputs_from_defaults(simulation_config, VoltageMetrics.NAME)
        if inputs.get("force_instantaneous", False):
            data = VoltageMetrics._get_required_exports_instantaneous(inputs)
        elif inputs.get("force_moving_average", False):
            data = VoltageMetrics._get_required_exports_moving_average(inputs, resolution)
        elif resolution >= timedelta(minutes=15):
            data = VoltageMetrics._get_required_exports_instantaneous(inputs)
        else:
            data = VoltageMetrics._get_required_exports_moving_average(inputs, resolution)

        return data

    @staticmethod
    def _get_required_exports_instantaneous(inputs):
        return {
            "Nodes": [
                {
                    "property": "VoltageMetrics",
                    "store_values_type": "all",
                    "limits": inputs["outer_limits"],
                },
            ]
        }

    @staticmethod
    def _get_required_exports_moving_average(inputs, resolution):
        window_size_td = timedelta(minutes=inputs["window_size_minutes"])
        if window_size_td % resolution != timedelta(0):
            raise InvalidConfiguration(
                f"window_size_minutes={window_size_td} must be a multiple of {resolution}"
            )
        window_size = int(window_size_td / resolution)

        return {
            "Nodes": [
                {
                    "property": "VoltageMetrics",
                    "store_values_type": "moving_average",
                    "limits": inputs["outer_limits"],
                    "window_size": window_size,
                },
                {
                    "property": "VoltageMetrics",
                    "store_values_type": "max",
                }
            ]
        }


class ReportGranularity(enum.Enum):
    """Specifies the granularity on which data is collected."""
    PER_ELEMENT_PER_TIME_POINT = "per_element_per_time_point"
    PER_ELEMENT_TOTAL = "per_element_total"
    ALL_ELEMENTS_PER_TIME_POINT = "all_elements_per_time_point"
    ALL_ELEMENTS_TOTAL = "all_elements_total"


REPORTS = {
    CapacitorStateChangeReport.NAME: CapacitorStateChangeReport,
    FeederLossesReport.NAME: FeederLossesReport,
    PvClippingReport.NAME: PvClippingReport,
    PvCurtailmentReport.NAME: PvCurtailmentReport,
    RegControlTapNumberChangeReport.NAME: RegControlTapNumberChangeReport,
    ThermalMetrics.NAME: ThermalMetrics,
    VoltageMetrics.NAME: VoltageMetrics,
}


def _get_inputs_from_defaults(simulation_config, name):
    options = _get_report_options(simulation_config, name)
    inputs = copy.deepcopy(getattr(REPORTS[name], "DEFAULTS"))
    for key, val in options.items():
        inputs[key] = val

    return inputs


def _get_report_options(simulation_config, name):
    for report in simulation_config["Reports"]["Types"]:
        if report["name"] == name:
            return report

    assert False, f"{name} is not present"
