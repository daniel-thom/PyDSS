
from collections import defaultdict
import abc
import copy
import logging
import time

import h5py
import numpy as np
import pandas as pd
import opendssdirect as dss

from PyDSS.common import DataConversion, StoreValuesType
from PyDSS.exceptions import InvalidConfiguration, InvalidParameter
from PyDSS.storage_filters import STORAGE_TYPE_MAP, StorageFilterBase
from PyDSS.value_storage import ValueByNumber


logger = logging.getLogger(__name__)


class Metric(abc.ABC):
    """Base class for all metrics"""
    def __init__(self, prop, dss_objs, options):
        self._name = prop.name
        self._base_path = None
        self._hdf_store = None
        self._max_chunk_bytes = options["Exports"]["HDF Max Chunk Bytes"]
        self._num_steps = None
        self._properties = {}  # StoreValuesType to ExportListProperty
        self._dss_objs = dss_objs

        self.add_property(prop)

    def add_property(self, prop):
        """Add an instance of ExportListProperty for tracking."""
        existing = self._properties.get(prop.store_values_type)
        if existing is None:
            self._properties[prop.store_values_type] = prop
        elif prop != existing:
            raise InvalidParameter(f"{prop.store_values_type} is already stored")

    @abc.abstractmethod
    def append_values(self, time_step):
        """Get the values for all elements at the current time step."""

    def close(self):
        """Perform any final writes to the container."""
        for container in self.iter_containers():
            if container is not None:
                container.close()

    def flush_data(self):
        """Flush any data in memory to storage."""
        for container in self.iter_containers():
            container.flush_data()

    def initialize_data_store(self, hdf_store, base_path, num_steps):
        """Initialize data store values."""
        self._hdf_store = hdf_store
        self._base_path = base_path
        self._num_steps = num_steps

    @staticmethod
    def is_circuit_wide():
        """Return True if this metric should be used once for a circuit."""
        return False

    @abc.abstractmethod
    def iter_containers(self):
        """Return an iterator over the StorageFilterBase containers."""

    @property
    def label(self):
        """Return a label for the metric.

        Returns
        -------
        str

        """
        prop = next(iter(self._properties.values()))
        return f"{prop.elem_class}.{prop.name}"

    def make_storage_container(self, hdf_store, path, prop, num_steps, max_chunk_bytes, values):
        """Make a storage container.

        Returns
        -------
        StorageFilterBase

        """
        if prop.store_values_type not in STORAGE_TYPE_MAP:
            raise InvalidConfiguration(f"unsupported {prop.store_values_type}")
        elem_names = [x.FullName for x in self._dss_objs]
        cls = STORAGE_TYPE_MAP[prop.store_values_type]
        container = cls(hdf_store, path, prop, num_steps, max_chunk_bytes, values, elem_names)
        return container

    def max_num_bytes(self):
        """Return the maximum number of bytes the containers could hold.

        Returns
        -------
        int

        """
        total = 0
        for container in self.iter_containers():
            if container is not None:
                total += container.max_num_bytes()
        return total


class ChangeCountMetric(Metric, abc.ABC):
    """Base class for any metric that only tracks number of changes."""
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._container = None
        self._last_values = {x.FullName: None for x in dss_objs}
        self._change_counts = {x.FullName: 0 for x in dss_objs}

    def append_values(self, time_step):
        pass

    def close(self):
        assert len(self._properties) == 1
        prop = next(iter(self._properties.values()))
        path = f"{self._base_path}/{prop.elem_class}/ElementProperties/{prop.storage_name}"
        values = [
            ValueByNumber(x, prop.name, y)
            for x, y in self._change_counts.items()
        ]
        # This class creates an instance of ValueContainer directly because
        # these metrics can only store one type, and so don't need an instance
        # of StorageFilterBase.
        self._container = StorageFilterBase.make_container(
            self._hdf_store,
            path,
            prop,
            self._num_steps,
            self._max_chunk_bytes,
            values,
            [x.FullName for x in self._dss_objs],
        )
        self._container.append(values)
        self._container.flush_data()

    def iter_containers(self):
        yield self._container


class MultiValueTypeMetrics(Metric, abc.ABC):
    """Stores a property with multiple values of StoreValueType.

    For example, a user might want to store a moving average as well as the
    max of all instantaneous values.

    """
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._containers = {}  # StoreValuesType to StorageFilterBase

    @abc.abstractmethod
    def _get_value(self, dss_obj, time_step):
        """Get a value at the current time step."""

    def _initialize_containers(self, values):
        prop_name = None
        for prop in self._properties.values():
            if prop_name is None:
                prop_name = prop.name
            else:
                assert prop.name == prop_name, f"{prop.name} {prop_name}"
            if prop.data_conversion != DataConversion.NONE:
                vals = [
                    convert_data(x.FullName, prop_name, y, prop.data_conversion)
                    for x, y in zip(self._dss_objs, values)
                ]
            else:
                vals = values
            path = f"{self._base_path}/{prop.elem_class}/ElementProperties/{prop.storage_name}"
            self._containers[prop.store_values_type] = self.make_storage_container(
                self._hdf_store,
                path,
                prop,
                self._num_steps,
                self._max_chunk_bytes,
                vals,
            )

    def append_values(self, time_step):
        start = time.time()
        values = [self._get_value(x, time_step) for x in self._dss_objs]

        if not self._containers:
            self._initialize_containers(values)

        for value_type, container in self._containers.items():
            prop = self._properties[value_type]
            if prop.data_conversion != DataConversion.NONE:
                vals = [
                    convert_data(x.FullName, prop.name, y, prop.data_conversion)
                    for x, y in zip(self._dss_objs, values)
                ]
            else:
                vals = values
            container.append_values(vals, time_step)

        return vals

    def iter_containers(self):
        return self._containers.values()


class OpenDssPropertyMetrics(MultiValueTypeMetrics):
    """Stores metrics for any OpenDSS element property."""

    def _get_value(self, dss_obj, _time_step):
        return dss_obj.UpdateValue(self._name)

    def append_values(self, time_step):
        curr_data = {}
        values = super().append_values(time_step)
        for dss_obj, value in zip(self._dss_objs, values):
            if len(value.make_columns()) > 1:
                for column, val in zip(value.make_columns(), value.value):
                    curr_data[column] = val
            else:
                curr_data[value.make_columns()[0]] = value.value

        return curr_data


class LineLoadingPercent(MultiValueTypeMetrics):
    """Calculates line loading percent at every time point."""
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._normal_amps = {}  # Name to normal_amps value

    def _get_value(self, dss_obj, _time_step):
        line = dss_obj
        normal_amps = self._normal_amps.get(line.Name)
        if normal_amps is None:
            normal_amps = line.GetValue("NormalAmps", convert=True).value
            self._normal_amps[line.Name] = normal_amps

        currents = line.UpdateValue("Currents").value
        current = max([abs(x) for x in currents])
        loading = current / normal_amps * 100
        return ValueByNumber(line.Name, "LineLoading", loading)


class TransformerLoadingPercent(MultiValueTypeMetrics):
    """Calculates transformer loading percent at every time point."""
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._normal_amps = {}  # Name to normal_amps value

    def _get_value(self, dss_obj, _time_step):
        transformer = dss_obj
        normal_amps = self._normal_amps.get(transformer.Name)
        if normal_amps is None:
            normal_amps = transformer.GetValue("NormalAmps", convert=True).value
            self._normal_amps[transformer.Name] = normal_amps

        currents = transformer.UpdateValue("Currents").value
        current = max([abs(x) for x in currents])
        loading = current / normal_amps * 100
        return ValueByNumber(transformer.Name, "TransformerLoading", loading)


class SummedElementsOpenDssPropertyMetric(Metric):
    """Sums all elements' values for a given property at each time point."""
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._container = None
        self._data_conversion = prop.data_conversion

    def append_values(self, time_step):
        total = None
        for obj in self._dss_objs:
            value = obj.UpdateValue(self._name)
            if self._data_conversion != DataConversion.NONE:
                value = convert_data(
                    "Total",
                    next(iter(self._properties.values())).name,
                    value,
                    self._data_conversion,
                )
            if total is None:
                total = value
            else:
                total += value

        if self._container is None:
            assert len(self._properties) == 1
            prop = list(self._properties.values())[0]
            assert prop.store_values_type in (StoreValuesType.ALL, StoreValuesType.SUM)
            total.set_name("Total")
            path = f"{self._base_path}/{prop.elem_class}/SummedElementProperties/{prop.storage_name}"
            self._container = self.make_storage_container(
                self._hdf_store,
                path,
                prop,
                self._num_steps,
                self._max_chunk_bytes,
                [total],
            )
        self._container.append_values([value], time_step)


    @staticmethod
    def is_circuit_wide():
        return True

    def iter_containers(self):
        yield self._container


class NodeVoltageMetrics(Metric):
    """Stores metrics for node voltages."""
    def __init__(self, prop, dss_obj, options):
        super().__init__(prop, dss_obj, options)
        # Indices for node names are tied to indices for node voltages.
        self._node_names = None
        self._containers = {}
        self._voltages = None

    def _make_values(self, voltages):
        return [ValueByNumber(x, "Voltage", y) for x, y in zip(self._node_names, voltages)]

    def append_values(self, time_step):
        voltages = dss.Circuit.AllBusMagPu()
        if not self._containers:
            # TODO: limit to objects that have been added
            self._node_names = dss.Circuit.AllNodeNames()
            self._voltages = [ValueByNumber(x, "Voltage", y) for x, y in zip(self._node_names, voltages)]
            for prop in self._properties.values():
                path = f"{self._base_path}/Nodes/ElementProperties/{prop.storage_name}"
                #values = self._make_values(voltages)
                self._containers[prop.store_values_type] = self.make_storage_container(
                    self._hdf_store,
                    path,
                    prop,
                    self._num_steps,
                    self._max_chunk_bytes,
                    self._voltages,
                )
        else:
            for i in range(len(voltages)):
                self._voltages[i].set_value_from_raw(voltages[i])
        for sv_type, prop in self._properties.items():
            self._containers[sv_type].append_values(self._voltages, time_step)

    @staticmethod
    def is_circuit_wide():
        return True

    def iter_containers(self):
        for sv_type in self._properties:
            if sv_type in self._containers:
                yield self._containers[sv_type]


class TrackCapacitorChangeCounts(ChangeCountMetric):
    """Store the number of changes for a capacitor."""

    def append_values(self, _time_step):
        for capacitor in self._dss_objs:
            self._update_counts(capacitor)

    def _update_counts(self, capacitor):
        dss.Capacitors.Name(capacitor.Name)
        if dss.CktElement.Name() != dss.Element.Name():
            raise InvalidParameter(
                f"Object is not a circuit element {capacitor.Name}"
            )
        states = dss.Capacitors.States()
        if states == -1:
            raise Exception(
                f"failed to get Capacitors.States() for {capacitor.Name}"
            )

        cur_value = sum(states)
        last_value = self._last_values[capacitor.FullName]
        if last_value is None and cur_value != last_value:
            logger.debug("%s changed state old=%s new=%s", capacitor.Name,
                         last_value, cur_value)
            self._change_counts[capacitor.FullName] += 1

        self._last_values[capacitor.FullName] = cur_value


class TrackRegControlTapNumberChanges(ChangeCountMetric):
    """Store the number of tap number changes for a RegControl."""

    def append_values(self, _time_step):
        for reg_control in self._dss_objs:
            self._update_counts(reg_control)

    def _update_counts(self, reg_control):
        dss.RegControls.Name(reg_control.Name)
        if reg_control.dss.CktElement.Name() != dss.Element.Name():
            raise InvalidParameter(
                f"Object is not a circuit element {reg_control.Name()}"
            )
        tap_number = dss.RegControls.TapNumber()
        last_value = self._last_values[reg_control.FullName]
        if last_value is not None:
            self._change_counts[reg_control.FullName] += abs(tap_number - last_value)
            logger.debug("%s changed count from %s to %s count=%s",
                         reg_control.Name, last_value, tap_number,
                         self._change_counts[reg_control.FullName])

        self._last_values[reg_control.FullName] = tap_number


class OpenDssExportMetric(Metric):
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        self._containers = {}
        filename = self._run_command()
        self.check_output(filename)

    def _run_command(self):
        cmd = f"{self.export_command()}"
        result = dss.utils.run_command(cmd)
        if not result:
            raise Exception(f"{cmd} failed")
        return result

    @abc.abstractmethod
    def check_output(self, filename):
        """Check that the output is expected."""

    @staticmethod
    @abc.abstractmethod
    def export_command(self):
        """Return the command to run in OpenDSS."""

    def iter_containers(self):
        yield None

    @abc.abstractmethod
    def parse_file(self, filename):
        """Parse data in filename."""

    @staticmethod
    def is_circuit_wide():
        return True

    def iter_containers(self):
        for sv_type in self._properties:
            if sv_type in self._containers:
                yield self._containers[sv_type]


class ExportPowersMetric(OpenDssExportMetric, abc.ABC):
    def __init__(self, prop, dss_objs, options):
        super().__init__(prop, dss_objs, options)
        # The OpenDSS file output upper-cases the name.
        # Make a mapping for fast matching and lookup.
        self._names = {}
        for i, dss_obj in enumerate(dss_objs):
            elem_type, name = dss_obj.FullName.split(".")
            self._names[f"{elem_type}.{name.upper()}"] = i

        # TODO: make this the full name
        self._powers =  [ValueByNumber(x, "Power", 0.0) for x in self._names]

    def append_values(self, time_step):
        filename = self._run_command()
        self.parse_file(filename)
        if not self._containers:
            for prop in self._properties.values():
                path = f"{self._base_path}/{prop.elem_class}/ElementProperties/{prop.storage_name}"
                self._containers[prop.store_values_type] = self.make_storage_container(
                    self._hdf_store,
                    path,
                    prop,
                    self._num_steps,
                    self._max_chunk_bytes,
                    self._powers,
                )

        for sv_type, prop in self._properties.items():
            self._containers[sv_type].append_values(self._powers, time_step)

    def check_output(self, filename):
        df = pd.read_csv(filename)
        expected = {0: "Element", 1: "Terminal", 2: "P(kW)"}
        for index, val in expected.items():
            if df.columns[index].strip() != val:
                raise Exception(f"Unexpected format in powers file: {index} {val} {df.columns}")

    @staticmethod
    def export_command():
        return "export powers"

    def parse_file(self, filename):
        with open(filename) as f_in:
            for line in f_in:
                fields = line.split(",")
                name = fields[0]
                if name in self._names:
                    #terminal = fields[1]
                    self._powers[self._names[name]].set_value_from_raw(float(fields[2]))


def convert_data(name, prop_name, value, conversion):
    if conversion == DataConversion.ABS:
        converted = copy.deepcopy(value)
        if isinstance(value.value, list):
            converted.set_value([abs(x) for x in value.value])
        else:
            converted.set_value(abs(value.value))
    elif conversion == DataConversion.SUM:
        converted = ValueByNumber(name, prop_name, sum(value.value))
    elif conversion == DataConversion.ABS_SUM:
        converted = ValueByNumber(name, prop_name, abs(sum(value.value)))
    else:
        converted = value

    return converted
