"""Provides access to PyDSS result data."""

from collections import defaultdict
import copy
import logging
import os
import re

import h5py
import numpy as np
import pandas as pd

from PyDSS.dataset_buffer import DatasetBuffer
from PyDSS.element_options import ElementOptions
from PyDSS.exceptions import InvalidParameter
from PyDSS.pydss_project import PyDssProject, RUN_SIMULATION_FILENAME
from PyDSS.reports import Reports, REPORTS, REPORTS_DIR
from PyDSS.utils.dataframe_utils import read_dataframe, write_dataframe
from PyDSS.utils.utils import dump_data, load_data
from PyDSS.value_storage import ValueStorageBase, DatasetPropertyType, \
    get_dataset_property_type, get_time_step_path


logger = logging.getLogger(__name__)


class PyDssResults:
    """Interface to perform analysis on PyDSS output data."""
    def __init__(
            self, project_path=None, project=None, in_memory=False,
            frequency=False, mode=False
        ):
        """Constructs PyDssResults object.

        Parameters
        ----------
        project_path : str | None
            Load project from files in path
        project : PyDssProject | None
            Existing project object
        in_memory : bool
            If true, load all exported data into memory.
        frequency : bool
            If true, add frequency column to all dataframes.
        mode : bool
            If true, add mode column to all dataframes.

        """
        options = ElementOptions()
        if project_path is not None:
            # TODO: handle old version?
            self._project = PyDssProject.load_project(
                project_path,
                simulation_file=RUN_SIMULATION_FILENAME,
            )
        elif project is None:
            raise InvalidParameter("project_path or project must be set")
        else:
            self._project = project
        self._fs_intf = self._project.fs_interface
        self._scenarios = []
        filename = self._project.get_hdf_store_filename()
        driver = "core" if in_memory else None
        self._hdf_store = h5py.File(filename, "r", driver=driver)

        if self._project.simulation_config["Exports"]["Log Results"]:
            for name in self._project.list_scenario_names():
                metadata = self._project.read_scenario_export_metadata(name)
                scenario_result = PyDssScenarioResults(
                    name,
                    self.project_path,
                    self._hdf_store,
                    self._fs_intf,
                    metadata,
                    options,
                    frequency=frequency,
                    mode=mode,
                )
                self._scenarios.append(scenario_result)

    def generate_reports(self):
        """Generate all reports specified in the configuration.

        Returns
        -------
        list
            list of report filenames

        """
        return Reports.generate_reports(self)

    def read_report(self, report_name):
        """Return the report data.

        Parameters
        ----------
        report_name : str

        Returns
        -------
        str

        """
        if report_name not in REPORTS:
            raise InvalidParameter(f"invalid report name {report_name}")
        report_cls = REPORTS[report_name]

        # This bypasses self._fs_intf because reports are always extracted.
        reports_dir = os.path.join(self._project.project_path, REPORTS_DIR)
        for filename in os.listdir(reports_dir):
            name, ext = os.path.splitext(filename)
            if name == os.path.splitext(report_cls.FILENAME)[0]:
                path = os.path.join(reports_dir, filename)
                if ext in (".json", ".toml"):
                    return load_data(path)
                if ext in (".csv", ".h5"):
                    return read_dataframe(path)

        raise InvalidParameter(f"did not find report {report_name} in {reports_dir}")

    @property
    def scenarios(self):
        """Return the PyDssScenarioResults instances for the project.

        Returns
        -------
        list
            list of PyDssScenarioResults

        """
        return self._scenarios

    def get_scenario(self, name):
        """Return the PyDssScenarioResults object for scenario with name.

        Parameters
        ----------
        name : str
            Scenario name

        Results
        -------
        PyDssScenarioResults

        Raises
        ------
        InvalidParameter
            Raised if the scenario does not exist.

        """
        for scenario in self._scenarios:
            if name == scenario.name:
                return scenario

        raise InvalidParameter(f"scenario {name} does not exist")

    @property
    def hdf_store(self):
        """Return a handle to the HDF data store.

        Returns
        -------
        h5py.File

        """
        return self._hdf_store

    @property
    def project_path(self):
        """Return the path to the PyDSS project.

        Returns
        -------
        str

        """
        return self._project.project_path

    def read_file(self, path):
        """Read a file from the PyDSS project.

        Parameters
        ----------
        path : str
            Path to the file relative from the project directory.

        Returns
        -------
        str
            Contents of the file

        """
        return self._fs_intf.read_file(path)

    @property
    def simulation_config(self):
        """Return the simulation configuration

        Returns
        -------
        dict

        """
        return self._project.simulation_config


class PyDssScenarioResults:
    """Contains results for one scenario."""
    def __init__(
            self, name, project_path, store, fs_intf, metadata, options,
            frequency=False, mode=False
        ):
        self._name = name
        self._project_path = project_path
        self._hdf_store = store
        self._metadata = metadata
        self._options = options
        self._fs_intf = fs_intf
        self._group = self._hdf_store[f"Exports/{name}"]
        self._elem_classes = [
            x for x in self._group if isinstance(self._group[x], h5py.Group)
        ]
        self._elems_by_class = defaultdict(set)
        self._elem_data_by_prop = defaultdict(dict)
        self._elem_nums_by_prop = defaultdict(dict)
        self._elem_indices_by_prop = defaultdict(dict)
        self._props_by_class = defaultdict(list)
        self._elem_props = defaultdict(list)
        self._column_ranges_per_elem = defaultdict(dict)
        #self._elem_prop_nums = defaultdict(dict)
        self._summed_elem_props = defaultdict(dict)
        self._summed_elem_timeseries_props = defaultdict(list)
        self._indices_df = None
        self._add_frequency = frequency
        self._add_mode = mode
        self._data_format_version = self._hdf_store.attrs["version"]

        self._parse_datasets()

    def _parse_datasets(self):
        for elem_class in self._elem_classes:
            class_group = self._group[elem_class]
            if "ElementProperties" in class_group:
                prop_group = class_group["ElementProperties"]
                for prop, dataset in prop_group.items():
                    dataset_property_type = get_dataset_property_type(dataset)
                    if dataset_property_type == DatasetPropertyType.TIME_STEP:
                        continue
                    if dataset_property_type == DatasetPropertyType.NUMBER:
                        self._elem_nums_by_prop[elem_class][prop] = []
                        prop_dict = self._elem_nums_by_prop
                    else:
                        assert dataset_property_type in (
                            DatasetPropertyType.ELEMENT_PROPERTY,
                            DatasetPropertyType.FILTERED,
                        )
                        self._elem_data_by_prop[elem_class][prop] = []
                        prop_dict = self._elem_data_by_prop

                    self._props_by_class[elem_class].append(prop)
                    self._elem_indices_by_prop[elem_class][prop] = {}
                    attrs = dataset.attrs
                    names = attrs["names"]
                    self._column_ranges_per_elem[elem_class][prop] = attrs["column_ranges_per_name"]
                    for i, name in enumerate(names):
                        self._elems_by_class[elem_class].add(name)
                        prop_dict[elem_class][prop].append(name)
                        self._elem_indices_by_prop[elem_class][prop][name] = i
                        self._elem_props[name].append(prop)
            else:
                self._elems_by_class[elem_class] = set()

            summed_elem_props = self._group[elem_class].get("SummedElementProperties", [])
            for prop in summed_elem_props:
                dataset = self._group[elem_class]["SummedElementProperties"][prop]
                dataset_property_type = get_dataset_property_type(dataset)
                if dataset_property_type == DatasetPropertyType.NUMBER:
                    df = DatasetBuffer.to_dataframe(dataset)
                    assert len(df) == 1
                    self._summed_elem_props[elem_class][prop] = {
                        x: df[x].values[0] for x in df.columns
                    }
                else:
                    self._summed_elem_timeseries_props[elem_class].append(prop)

    @property
    def name(self):
        """Return the name of the scenario.

        Returns
        -------
        str

        """
        return self._name

    def export_data(self, path=None, fmt="csv", compress=False):
        """Export data to path.

        Parameters
        ----------
        path : str
            Output directory; defaults to scenario exports path
        fmt : str
            Filer format type (csv, h5)
        compress : bool
            Compress data

        """
        if path is None:
            path = os.path.join(self._project_path, "Exports", self._name)
        os.makedirs(path, exist_ok=True)

        #for elem_class in self.list_element_classes():
        #    for prop in self.list_element_properties(elem_class):
        #        try:
        #            df = self.get_full_dataframe(elem_class, prop)
        #        except InvalidParameter:
        #            logger.info(f"cannot create full dataframe for %s %s", elem_class, prop)
        #            self._export_filtered_dataframes(elem_class, prop, path, fmt, compress)
        #            continue
        #        base = "__".join([elem_class, prop])
        #        filename = os.path.join(path, base + "." + fmt.replace(".", ""))
        #        write_dataframe(df, filename, compress=compress)

        # TODO DT
        #if self._elem_prop_nums:
        #    data = copy.deepcopy(self._elem_prop_nums)
        #    for elem_class, prop, name, val in self.iterate_element_property_numbers():
        #        # JSON lib cannot serialize complex numbers.
        #        if isinstance(val, np.ndarray):
        #            new_val = []
        #            convert_str = val.dtype == "complex"
        #            for item in val:
        #                if convert_str:
        #                    item = str(item)
        #                new_val.append(item)
        #            data[elem_class][prop][name] = new_val
        #        elif isinstance(val, complex):
        #            data[elem_class][prop][name] = str(val)

        #    filename = os.path.join(path, "element_property_numbers.json")
        #    dump_data(data, filename, indent=2)

        # TODO DT: export summed elements
        logger.info("Exported data to %s", path)

    def _export_filtered_dataframes(self, elem_class, prop, path, fmt, compress):
        for name, df in self.iterate_dataframes(elem_class, prop):
            base = "__".join([elem_class, prop, name])
            filename = os.path.join(path, base + "." + fmt.replace(".", ""))
            write_dataframe(df, filename, compress=compress)

    def get_dataframe(self, element_class, prop, element_name, real_only=False, abs_val=False, **kwargs):
        """Return the dataframe for an element.

        Parameters
        ----------
        element_class : str
        prop : str
        element_name : str
        real_only : bool
            If dtype of any column is complex, drop the imaginary component.
        abs_val : bool
            If dtype of any column is complex, compute its absolute value.
        kwargs : **kwargs
            Filter on options. Option values can be strings or regular expressions.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        InvalidParameter
            Raised if the element is not stored.

        """
        if element_name not in self._elem_props:
            raise InvalidParameter(f"element {element_name} is not stored")

        dataset = self._group[f"{element_class}/ElementProperties/{prop}"]
        prop_type = get_dataset_property_type(dataset)
        if prop_type == DatasetPropertyType.ELEMENT_PROPERTY:
            return self._get_elem_prop_dataframe(element_class, prop, element_name, dataset, real_only=real_only, abs_val=abs_val, **kwargs)
        elif prop_type == DatasetPropertyType.FILTERED:
            return self._get_filtered_dataframe(element_class, prop, element_name, dataset, real_only=real_only, abs_val=abs_val, **kwargs)
        else:
            assert False, str(prop_type)

    def get_filtered_dataframes(self, element_class, prop, real_only=False, abs_val=False):
        """Return the dataframes for all elements.

        Calling this is much more efficient than calling get_dataframe for each
        element.

        Parameters
        ----------
        element_class : str
        prop : str
        element_name : str
        real_only : bool
            If dtype of any column is complex, drop the imaginary component.
        abs_val : bool
            If dtype of any column is complex, compute its absolute value.

        Returns
        -------
        dict
            key = str (name), val = pd.DataFrame

        """
        dataset = self._group[f"{element_class}/ElementProperties/{prop}"]
        columns = dataset.attrs["columns"]
        names = dataset.attrs["names"]
        length = dataset.attrs["length"]
        indices_df = self._get_indices_df()
        data_vals = dataset[:]
        elem_data = defaultdict(list)
        elem_timestamps = defaultdict(list)

        # The time_step_dataset has these columns:
        # 1. time step index
        # 2. element index
        # Each row describes the source data in the dataset row.
        path = dataset.attrs["time_step_path"]
        time_step_data = self._hdf_store[path][:]

        assert length == self._hdf_store[path].attrs["length"]
        data = []
        timestamps = []
        for i in range(length):
            ts_index = time_step_data[:, 0][i]
            elem_index = time_step_data[:, 1][i]
            # TODO DT: more than one column?
            val = data_vals[i, 0]
            if real_only:
                val = val.real
            elif abs_val:
                val = abs(val)
            elem_data[elem_index].append(val)
            elem_timestamps[elem_index].append(indices_df.iloc[ts_index, 0])

        dfs = {}
        for elem_index, vals in elem_data.items():
            elem_name = names[elem_index]
            cols = self._fix_columns(elem_name, columns)
            dfs[elem_name] = pd.DataFrame(
                vals,
                columns=cols,
                index=elem_timestamps[elem_index],
            )
        return dfs

    def get_summed_element_total(self, element_class, prop):
        """Return the total value for a summed element property.

        Parameters
        ----------
        element_class : str
        prop : str

        Returns
        -------
        dict

        Raises
        ------
        InvalidParameter
            Raised if the element class is not stored.

        """
        if element_class not in self._summed_elem_props:
            raise InvalidParameter(f"{element_class} is not stored")
        if prop not in self._summed_elem_props[element_class]:
            raise InvalidParameter(f"{prop} is not stored")

        return self._summed_elem_props[element_class][prop]

    @property
    def element_property_numbers(self):
        """Return all element property values stored as numbers.

        Returns
        -------
        dict

        """
        return self._elem_prop_nums

    def get_element_property_number(self, element_class, prop, element_name):
        """Return the number stored for the element property."""
        if element_class not in self._elem_nums_by_prop:
            raise InvalidParameter(f"{element_class} is not stored")
        if prop not in self._elem_nums_by_prop[element_class]:
            raise InvalidParameter(f"{prop} is not stored")
        if element_name not in self._elem_nums_by_prop[element_class][prop]:
            raise InvalidParameter(f"{element_name} is not stored")
        dataset = self._group[f"{element_class}/ElementProperties/{prop}"]
        col_range = self._get_element_column_range(element_class, prop, element_name)
        start = col_range[0]
        length = col_range[1]
        if length == 1:
            return dataset[:][0][start]
        return dataset[:][0][start: start + length]

    def get_full_dataframe(self, element_class, prop, real_only=False, abs_val=False):
        """Return a dataframe containing all data.  The dataframe is copied.

        Parameters
        ----------
        element_class : str
        prop : str
        real_only : bool
            If dtype of any column is complex, drop the imaginary component.
        abs_val : bool
            If dtype of any column is complex, compute its absolute value.

        Returns
        -------
        pd.DataFrame

        """
        if prop not in self.list_element_properties(element_class):
            raise InvalidParameter(f"property {prop} is not stored")

        dataset = self._group[f"{element_class}/ElementProperties/{prop}"]
        df = DatasetBuffer.to_dataframe(dataset)
        self._finalize_dataframe(df, dataset, real_only=real_only, abs_val=abs_val)
        return df

    def get_option_values(self, element_class, prop, element_name):
        """Return the option values for the element property.

        element_class : str
        prop : str
        element_name : str

        Returns
        -------
        list

        """
        df = self.get_dataframe(element_class, prop, element_name)
        return ValueStorageBase.get_option_values(df, element_name)

    def get_summed_element_dataframe(self, element_class, prop, real_only=False, abs_val=False):
        """Return the dataframe for a summed element property.

        Parameters
        ----------
        element_class : str
        prop : str
        real_only : bool
            If dtype of any column is complex, drop the imaginary component.
        abs_val : bool
            If dtype of any column is complex, compute its absolute value.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        InvalidParameter
            Raised if the element class is not stored.

        """
        if element_class not in self._summed_elem_timeseries_props:
            raise InvalidParameter(f"{element_class} is not stored")
        if prop not in self._summed_elem_timeseries_props[element_class]:
            raise InvalidParameter(f"{prop} is not stored")

        elem_group = self._group[element_class]["SummedElementProperties"]
        dataset = elem_group[prop]
        df = DatasetBuffer.to_dataframe(dataset)
        self._add_indices_to_dataframe(df)

        if real_only:
            for column in df.columns:
                if df[column].dtype == np.complex:
                    df[column] = [x.real for x in df[column]]
        elif abs_val:
            for column in df.columns:
                if df[column].dtype == np.complex:
                    df[column] = [abs(x) for x in df[column]]

        return df

    def iterate_dataframes(self, element_class, prop, real_only=False, abs_val=False, **kwargs):
        """Returns a generator over the dataframes by element name.

        Parameters
        ----------
        element_class : str
        prop : str
        real_only : bool
            If dtype of any column is complex, drop the imaginary component.
        abs_val : bool
            If dtype of any column is complex, compute its absolute value.
        kwargs : **kwargs
            Filter on options. Option values can be strings or regular expressions.

        Returns
        -------
        tuple
            Tuple containing the name or property and a pd.DataFrame

        """
        for name in self.list_element_names(element_class):
            if prop in self._elem_props[name]:
                df = self.get_dataframe(
                    element_class, prop, name, real_only=real_only, abs_val=abs_val, **kwargs
                )
                yield name, df

    def iterate_element_property_numbers(self):
        """Return a generator over all element properties stored as numbers.

        Yields
        ------
        tuple
            element_class, property, element_name, value

        """
        for elem_class in self._elem_prop_nums:
            for prop in self._elem_prop_nums[elem_class]:
                for name, val in self._elem_prop_nums[elem_class][prop].items():
                    yield elem_class, prop, name, val

    def list_element_classes(self):
        """Return the element classes stored in the results.

        Returns
        -------
        list

        """
        return self._elem_classes[:]

    def list_element_names(self, element_class, prop=None):
        """Return the element names for a property stored in the results.

        Parameters
        ----------
        element_class : str
        prop : str

        Returns
        -------
        list

        """
        # TODO: prop is deprecated
        return self._elems_by_class.get(element_class, [])

    def list_element_properties(self, element_class, element_name=None):
        """Return the properties stored in the results for a class.

        Parameters
        ----------
        element_class : str
        element_name : str | None
            If not None, list properties only for that name.

        Returns
        -------
        list

        Raises
        ------
        InvalidParameter
            Raised if the element_class is not stored.

        """
        if element_class not in self._props_by_class:
            raise InvalidParameter(f"class={element_class} is not stored")
        if element_name is None:
            return sorted(list(self._props_by_class[element_class]))
        return self._elem_props.get(element_name, [])

    def list_element_property_numbers(self, element_class, element_name):
        nums = []
        for elem_class in self._elem_prop_nums:
            for prop in self._elem_prop_nums[elem_class]:
                for name in self._elem_prop_nums[elem_class][prop]:
                    if name == element_name:
                        nums.append(prop)
        return nums

    def list_element_property_options(self, element_class, prop):
        """List the possible options for the element class and property.

        Parameters
        ----------
        element_class : str
        prop : str

        Returns
        -------
        list

        """
        return self._options.list_options(element_class, prop)

    def list_element_info_files(self):
        """Return the files describing the OpenDSS element objects.

        Returns
        -------
        list
            list of filenames (str)

        """
        return self._metadata["element_info_files"]

    def list_summed_element_properties(self, element_class):
        """Return the properties stored for a class where the values are a sum
        of all elements.

        Parameters
        ----------
        element_class : str

        Returns
        -------
        list

        Raises
        ------
        InvalidParameter
            Raised if the element_class is not stored.

        """
        if element_class not in self._summed_elem_props:
            raise InvalidParameter(f"class={element_class} is not stored")
        return self._summed_elem_props[element_class]

    def read_element_info_file(self, filename):
        """Return the contents of file describing an OpenDSS element object.

        Parameters
        ----------
        filename : str
            full path to a file (returned by list_element_info_files) or
            an element class, like "Transformers"

        Returns
        -------
        pd.DataFrame

        """
        if "." not in filename:
            actual = None
            for _file in self.list_element_info_files():
                basename = os.path.splitext(os.path.basename(_file))[0]
                if basename.replace("Info", "") == filename:
                    actual = _file
            if actual is None:
                raise InvalidParameter(
                    f"element info file for {filename} is not stored"
                )
            filename = actual

        return self._fs_intf.read_csv(filename)

    def read_capacitor_changes(self):
        """Read the capacitor state changes from the OpenDSS event log.

        Returns
        -------
        dict
            Maps capacitor names to count of state changes.

        """
        text = self.read_file(self._metadata["event_log"])
        return _read_capacitor_changes(text)

    def read_event_log(self):
        """Returns the event log for the scenario.

        Returns
        -------
        list
            list of dictionaries (one dict for each row in the file)

        """
        text = self.read_file(self._metadata["event_log"])
        return _read_event_log(text)

    def read_pv_profiles(self):
        """Returns exported PV profiles for all PV systems.

        Returns
        -------
        dict

        """
        return self._fs_intf.read_scenario_pv_profiles(self._name)

    def _check_options(self, element_class, prop, **kwargs):
        """Checks that kwargs are valid and returns available option names."""
        for option in kwargs:
            if not self._options.is_option_valid(element_class, prop, option):
                raise InvalidParameter(
                    f"class={element_class} property={prop} option={option} is invalid"
                )

        return self._options.list_options(element_class, prop)

    def read_file(self, path):
        """Read a file from the PyDSS project.

        Parameters
        ----------
        path : str
            Path to the file relative from the project directory.

        Returns
        -------
        str
            Contents of the file

        """
        return self._fs_intf.read_file(path)

    def _add_indices_to_dataframe(self, df):
        indices_df = self._get_indices_df()
        df["Timestamp"] = indices_df["Timestamp"]
        if self._add_frequency:
            df["Frequency"] = indices_df["Frequency"]
        if self._add_mode:
            df["Simulation Mode"] = indices_df["Simulation Mode"]
        df.set_index("Timestamp", inplace=True)

    def _finalize_dataframe(self, df, dataset, real_only=False, abs_val=False):
        dataset_property_type = get_dataset_property_type(dataset)
        if dataset_property_type == DatasetPropertyType.FILTERED:
            time_step_path = get_time_step_path(dataset)
            time_step_dataset = self._hdf_store[time_step_path]
            df["TimeStep"] = DatasetBuffer.to_datetime(time_step_dataset)
            df.set_index("TimeStep", inplace=True)
        else:
            self._add_indices_to_dataframe(df)

        if real_only:
            for column in df.columns:
                if df[column].dtype == np.complex:
                    df[column] = [x.real for x in df[column]]
        elif abs_val:
            for column in df.columns:
                if df[column].dtype == np.complex:
                    df[column] = [abs(x) for x in df[column]]

    @staticmethod
    def _fix_columns(name, columns):
        cols = []
        for column in columns:
            fields = column.split(ValueStorageBase.DELIMITER)
            fields[0] = name
            cols.append(ValueStorageBase.DELIMITER.join(fields))
        return cols

    def _get_elem_prop_dataframe(self, elem_class, prop, name, dataset, real_only=False, abs_val=False, **kwargs):
        col_range = self._get_element_column_range(elem_class, prop, name)
        df = DatasetBuffer.to_dataframe(dataset, column_range=col_range)

        if kwargs:
            options = self._check_options(elem_class, prop, **kwargs)
            columns = ValueStorageBase.get_columns(df, name, options, **kwargs)
            df = df[columns]

        self._finalize_dataframe(df, dataset, real_only=real_only, abs_val=abs_val)
        return df

    def _get_element_column_range(self, elem_class, prop, name):
        elem_index = self._elem_indices_by_prop[elem_class][prop][name]
        col_range = self._column_ranges_per_elem[elem_class][prop][elem_index]
        return col_range

    def _get_filtered_dataframe(self, elem_class, prop, name, dataset, real_only=False, abs_val=False, **kwargs):
        indices_df = self._get_indices_df()
        elem_index = self._elem_indices_by_prop[elem_class][prop][name]
        length = dataset.attrs["length"]
        data_vals = dataset[:]

        # The time_step_dataset has these columns:
        # 1. time step index
        # 2. element index
        # Each row describes the source data in the dataset row.
        path = dataset.attrs["time_step_path"]
        time_step_data = self._hdf_store[path][:]

        assert length == self._hdf_store[path].attrs["length"]
        data = []
        timestamps = []
        for i in range(length):
            stored_elem_index = time_step_data[:, 1][i]
            if stored_elem_index == elem_index:
                ts_index = time_step_data[:, 0][i]
                # TODO DT: more than one column?
                val = data_vals[i, 0]
                # TODO: profile this vs a df operation at end
                if real_only:
                    val = val.real
                elif abs_val:
                    val = abs(val)
                data.append(val)
                timestamps.append(indices_df.iloc[ts_index, 0])

        columns = self._fix_columns(name, dataset.attrs["columns"])
        return pd.DataFrame(data, columns=columns, index=timestamps)

    def _get_indices_df(self):
        if self._indices_df is None:
            self._make_indices_df()
        return self._indices_df

    def _make_indices_df(self):
        data = {"Timestamp": self._group["Timestamp"][:, 0]}
        if self._add_frequency:
            data["Frequency"] = self._group["Frequency"][:, 0]
        if self._add_mode:
            data["Simulation Mode"] = self._group["Mode"][:, 0]
        df = pd.DataFrame(data)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        self._indices_df = df


def _read_capacitor_changes(event_log_text):
    """Read the capacitor state changes from an OpenDSS event log.

    Parameters
    ----------
    event_log_text : str
        Text of event log

    Returns
    -------
    dict
        Maps capacitor names to count of state changes.

    """
    capacitor_changes = {}
    regex = re.compile(r"(Capacitor\.\w+)")

    data = _read_event_log(event_log_text)
    for row in data:
        match = regex.search(row["Element"])
        if match:
            name = match.group(1)
            if name not in capacitor_changes:
                capacitor_changes[name] = 0
            action = row["Action"].replace("*", "")
            if action in ("OPENED", "CLOSED", "STEP UP"):
                capacitor_changes[name] += 1

    return capacitor_changes


def _read_event_log(event_log_text):
    """Return OpenDSS event log information.

    Parameters
    ----------
    event_log_text : str
        Text of event log


    Returns
    -------
    list
        list of dictionaries (one dict for each row in the file)

    """
    data = []
    if not event_log_text:
        return data

    for line in event_log_text.split("\n"):
        if line == "":
            continue
        tokens = [x.strip() for x in line.split(",")]
        row = {}
        for token in tokens:
            name_and_value = [x.strip() for x in token.split("=")]
            name = name_and_value[0]
            value = name_and_value[1]
            row[name] = value
        data.append(row)

    return data
