
from collections import defaultdict
import abc
import copy
import logging
import time

import h5py
import numpy as np
import opendssdirect as dss

from PyDSS.common import DataConversion, StoreValuesType
from PyDSS.exceptions import InvalidConfiguration, InvalidParameter
from PyDSS.utils.simulation_utils import CircularBufferHelper
from PyDSS.value_storage import ValueContainer, ValueByNumber


logger = logging.getLogger(__name__)


class StorageBase(abc.ABC):
    """Base class for storage containers"""

    def __init__(self, hdf_store, path, prop, num_steps, max_chunk_bytes, value):
        self._prop = prop
        self._container = self.make_container(
            hdf_store,
            path,
            prop,
            num_steps,
            max_chunk_bytes,
            value,
        )

    @abc.abstractmethod
    def append_value(self, value, timestamp):
        """Store a new value."""

    def close(self):
        """Perform any final writes to the container."""
        self.flush_data()

    def flush_data(self):
        """Flush data to disk."""
        self._container.flush_data()

    def max_num_bytes(self):
        """Return the maximum number of bytes the container could hold.

        Returns
        -------
        int

        """
        return self._container.max_num_bytes()

    @staticmethod
    def make_container(hdf_store, path, prop, num_steps, max_chunk_bytes, value):
        """Return an instance of ValueContainer for storing values."""
        container = ValueContainer(
            value,
            hdf_store,
            path,
            prop.get_max_size(num_steps),
            dataset_property_type=prop.get_dataset_property_type(),
            max_chunk_bytes=max_chunk_bytes,
            store_timestamp=prop.should_store_timestamp(),
        )
        logger.debug("Created storage container path=%s", path)
        return container


class StorageAll(StorageBase):
    """Store values at every time point, optionally filtered."""

    def append_value(self, value, timestamp):
        if self._prop.should_store_value(value.value):
            self._container.append(value, timestamp=timestamp)


"""
class StorageChangeCount(StorageBase):
    def __init__(self, *args):
        super().__init__(*args)
        self._last_value = None
        self._change_count = (None, 0)

    def append_value(self, value, timestamp):
        assert False
"""


class StorageMax(StorageBase):
    """Stores the max value across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._max = None

    def append_value(self, value, timestamp):
        self._handle_value(value)

    def close(self):
        self._container.append(self._max)
        self._container.flush_data()

    def _handle_value(self, value):
        if (self._max is None or value > self._max) and not np.isnan(value.value):
            self._max = value


class StorageMovingAverage(StorageBase):
    """Stores a moving average across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._buf = CircularBufferHelper(self._prop)

    def append_value(self, value, timestamp):
        # Store every value in the circular buffer. Apply limits to the
        # moving average.
        self._buf.append(value.value)
        avg = self._buf.average()
        if self._prop.should_store_value(avg):
            # TODO: perf issue?
            new_value = copy.deepcopy(value)
            new_value.set_element_property(self._prop.storage_name)
            new_value.set_value(avg)
            self._container.append(new_value, timestamp=timestamp)


class StorageMovingAverageMax(StorageMax):
    """Stores the max value of a moving average across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._buf = CircularBufferHelper(self._prop)

    def append_value(self, value, timestamp):
        self._buf.append(value.value)
        avg = self._buf.average()
        logger.info("append avg value value=%s avg=%s", value.value, avg)
        new_value = copy.deepcopy(value)
        new_value.set_element_property(self._prop.storage_name)
        new_value.set_value(avg)
        self._handle_value(new_value)


class StorageSum(StorageBase):
    """Keeps a running sum of all values and records the total."""
    def __init__(self, *args):
        super().__init__(*args)
        self._sum = None

    def append_value(self, value, _timestamp):
        if self._sum is None:
            self._sum = value
        else:
            self._sum += value

    def close(self):
        assert self._sum is not None
        self._container.append(self._sum)
        self.flush_data()


STORAGE_TYPE_MAP = {
    StoreValuesType.ALL: StorageAll,
    #StoreValuesType.CHANGE_COUNT: StorageChangeCount,
    StoreValuesType.MAX: StorageMax,
    StoreValuesType.MOVING_AVERAGE: StorageMovingAverage,
    StoreValuesType.MOVING_AVERAGE_MAX: StorageMovingAverageMax,
    StoreValuesType.SUM: StorageSum,
}


class Metric(abc.ABC):
    """Base class for all metrics"""
    def __init__(self, prop, dss_obj, options):
        self._name = prop.name
        self._base_path = None
        self._hdf_store = None
        self._max_chunk_bytes = options["Exports"]["HDF Max Chunk Bytes"]
        self._num_steps = None
        self._properties = {}  # StoreValuesType to ExportListProperty
        self._dss_objs = [dss_obj]

        self.add_property(prop)

    def add_dss_obj(self, dss_obj):
        """Add an instance of dssObjectBase for tracking in this metric."""
        self._dss_objs.append(dss_obj)

    def add_property(self, prop):
        """Add an instance of ExportListProperty for tracking."""
        existing = self._properties.get(prop.store_values_type)
        if existing is None:
            self._properties[prop.store_values_type] = prop
        elif prop != existing:
            raise InvalidParameter(f"{prop.store_values_type} is already stored")

    @abc.abstractmethod
    def append_values(self, timestamp):
        """Get the value at the current timestamp."""

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
    def make_storage_container(hdf_store, path, prop, num_steps, max_chunk_bytes, value):
        """Make a storage container.

        Returns
        -------
        StorageBase

        """
        if prop.store_values_type not in STORAGE_TYPE_MAP:
            raise InvalidConfiguration(f"unsupported {prop.store_values_type}")
        cls = STORAGE_TYPE_MAP[prop.store_values_type]
        container = cls(hdf_store, path, prop, num_steps, max_chunk_bytes, value)
        return container

    @staticmethod
    def is_circuit_wide():
        """Return True if this metric should be used once for a circuit."""
        return False

    @abc.abstractmethod
    def iter_containers(self):
        """Return an iterator over the StorageBase containers."""

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
    def __init__(self, prop, dss_obj, options):
        super().__init__(prop, dss_obj, options)
        self._container = None
        self._last_value = None
        self._change_count = 0

    def append_values(self, timestamp):
        pass

    def close(self):
        assert len(self._properties) == 1
        prop = list(self._properties.values())[0]
        assert len(self._dss_objs) == 1
        obj = self._dss_objs[0]
        path = f"{self._base_path}/{prop.elem_class}/Elements/{obj.FullName}/{prop.storage_name}"
        value = ValueByNumber(obj.FullName, prop.name, self._change_count)
        # This class creates an instance of ValueContainer directly because
        # these metrics can only store one type, and so don't need an instance
        # of StorageBase.
        self._container = StorageBase.make_container(
            self._hdf_store,
            path,
            prop,
            self._num_steps,
            self._max_chunk_bytes,
            value,
        )
        self._container.append(value)
        self._container.flush_data()

    def iter_containers(self):
        yield self._container


class MultiValueTypeMetrics(Metric, abc.ABC):
    """Stores a property with multiple values of StoreValueType.

    For example, a user might want to store a moving average as well as the
    max of all instantaneous values.

    """
    def __init__(self, prop, dss_obj, options):
        super().__init__(prop, dss_obj, options)
        self._containers = {}  # StoreValuesType to StorageBase

    @abc.abstractmethod
    def _get_value(self, timestamp):
        """Get a value at the current timestamp."""

    def append_values(self, timestamp):
        start = time.time()
        value = self._get_value(timestamp)

        if not self._containers:
            assert len(self._dss_objs) == 1, self._dss_objs
            obj = self._dss_objs[0]
            prop_name = None
            for prop in self._properties.values():
                if prop_name is None:
                    prop_name = prop.name
                else:
                    assert prop.name == prop_name, f"{prop.name} {prop_name}"
                if prop.data_conversion != DataConversion.NONE:
                    val = convert_data(obj.FullName, prop_name, value, prop.data_conversion)
                else:
                    val = value
                path = f"{self._base_path}/{prop.elem_class}/Elements/{obj.FullName}/{prop.storage_name}"
                self._containers[prop.store_values_type] = self.make_storage_container(
                    self._hdf_store,
                    path,
                    prop,
                    self._num_steps,
                    self._max_chunk_bytes,
                    val,
                )

        for value_type, container in self._containers.items():
            prop = self._properties[value_type]
            if prop.data_conversion != DataConversion.NONE:
                val = convert_data(
                    self._dss_objs[0].FullName,
                    prop.name,
                    value,
                    prop.data_conversion,
                )
            else:
                val = value
            container.append_value(val, timestamp)

        return val

    def iter_containers(self):
        return self._containers.values()


class OpenDssPropertyMetrics(MultiValueTypeMetrics):
    """Stores metrics for any OpenDSS element property."""

    def _get_value(self, _timestamp):
        return self._dss_objs[0].GetValue(self._name, convert=True)

    def append_values(self, timestamp):
        curr_data = {}
        value = super().append_values(timestamp)
        if len(value.make_columns()) > 1:
            for column, val in zip(value.make_columns(), value.value):
                curr_data[column] = val
        else:
            curr_data[value.make_columns()[0]] = value.value

        return curr_data


class LineLoadingPercent(MultiValueTypeMetrics):
    """Calculates line loading percent at every time point."""

    def _get_value(self, _timestamp):
        line = self._dss_objs[0]
        normal_amps = line.GetValue("NormalAmps", convert=True).value
        currents = line.GetValue("Currents", convert=True).value
        current = max([abs(x) for x in currents])
        loading = current / normal_amps * 100
        return ValueByNumber(line.Name, "LineLoading", loading)


class TransformerLoadingPercent(MultiValueTypeMetrics):
    """Calculates transformer loading percent at every time point."""

    def _get_value(self, _timestamp):
        transformer = self._dss_objs[0]
        normal_amps = transformer.GetValue("NormalAmps", convert=True).value
        currents = transformer.GetValue("Currents", convert=True).value
        current = max([abs(x) for x in currents])
        loading = current / normal_amps * 100
        return ValueByNumber(transformer.Name, "TransformerLoading", loading)


class SummedElementsOpenDssPropertyMetric(Metric):
    """Sums all elements' values for a given property at each time point."""
    def __init__(self, prop, dss_obj, options):
        super().__init__(prop, dss_obj, options)
        self._container = None
        self._data_conversion = prop.data_conversion

    def append_values(self, timestamp):
        start = time.time()
        total = None
        for obj in self._dss_objs:
            value = obj.GetValue(self._name, convert=True)
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
                total,
            )
        self._container.append_value(value, timestamp)


    @staticmethod
    def is_circuit_wide():
        return True

    def iter_containers(self):
        yield self._container


class NodeVoltageMetrics(Metric):
    """Stores metrics for node voltages."""
    def __init__(self, prop, dss_obj, options):
        super().__init__(prop, dss_obj, options)
        self._step_number = 1
        # Indices for node names are tied to indices for node voltages.
        self._node_names = None
        self._containers = defaultdict(dict)

    def _iter_items(self):
        for prop in self._properties.values():
            for i, node_name in enumerate(self._node_names):
                yield i, node_name, prop

    @staticmethod
    def _make_value(name, value):
        return ValueByNumber(name, "Voltage", value)

    def append_values(self, timestamp):
        start = time.time()
        voltages = dss.Circuit.AllBusMagPu()
        if not self._containers:
            # TODO: limit to objects that have been added
            self._node_names = dss.Circuit.AllNodeNames()
            for i, node_name, prop in self._iter_items():
                path = f"{self._base_path}/Nodes/Elements/{node_name}/{prop.storage_name}"
                value = self._make_value(node_name, voltages[i])
                self._containers[prop.store_values_type][node_name] = self.make_storage_container(
                    self._hdf_store,
                    path,
                    prop,
                    self._num_steps,
                    self._max_chunk_bytes,
                    value,
                )

        for i, node_name, prop in self._iter_items():
            value = self._make_value(node_name, voltages[i])
            self._containers[prop.store_values_type][node_name].append_value(value, timestamp)

        self._step_number += 1

    @staticmethod
    def is_circuit_wide():
        return True

    def iter_containers(self):
        for _, node_name, prop in self._iter_items():
            yield self._containers[prop.store_values_type][node_name]


class TrackCapacitorChangeCounts(ChangeCountMetric):
    """Store the number of changes for a capacitor."""

    def append_values(self, timestamp):
        start = time.time()
        capacitor = self._dss_objs[0]
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
        if self._last_value is None and cur_value != self._last_value:
            logger.debug("%s changed state old=%s new=%s", capacitor.Name,
                         self._last_value, cur_value)
            self._change_count += 1

        self._last_value = cur_value


class TrackRegControlTapNumberChanges(ChangeCountMetric):
    """Store the number of tap number changes for a RegControl."""

    def append_values(self, timestamp):
        start = time.time()
        reg_control = self._dss_objs[0]
        dss.RegControls.Name(reg_control.Name)
        if reg_control.dss.CktElement.Name() != dss.Element.Name():
            raise InvalidParameter(
                f"Object is not a circuit element {reg_control.Name()}"
            )
        tap_number = dss.RegControls.TapNumber()
        if self._last_value is not None:
            self._change_count += abs(tap_number - self._last_value)
            logger.debug("%s changed count from %s to %s count=%s",
                         reg_control.Name, self._last_value, tap_number,
                         self._change_count)

        self._last_value = tap_number


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
