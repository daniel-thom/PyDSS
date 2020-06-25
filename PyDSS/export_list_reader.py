from collections import defaultdict, namedtuple
import enum
import logging
import os
import re

from PyDSS.common import DataConversion, LimitsFilter, StoreValuesType
from PyDSS.pyContrReader import pyExportReader
from PyDSS.utils.simulation_utils import calculate_line_loading_percent, \
    calculate_transformer_loading_percent, track_capacitor_state_changes, \
    track_reg_control_tap_number_changes
from PyDSS.utils.utils import load_data
from PyDSS.exceptions import InvalidConfiguration, InvalidParameter
from PyDSS.metrics import NodeVoltageMetrics, TrackCapacitorChangeCounts, \
    TrackRegControlTapNumberChanges, LineLoadingPercent, \
    TransformerLoadingPercent
from PyDSS.value_storage import DatasetPropertyType


MinMax = namedtuple("MinMax", "min, max")


CUSTOM_METRICS = {
    "Capacitors.TrackStateChanges": TrackCapacitorChangeCounts,
    "Lines.LoadingPercent": LineLoadingPercent,
    "RegControls.TrackTapNumberChanges": TrackRegControlTapNumberChanges,
    "Transformers.LoadingPercent": TransformerLoadingPercent,
    "Nodes.VoltageMetrics": NodeVoltageMetrics,
}

logger = logging.getLogger(__name__)


class ExportListProperty:
    """Contains export options for an element property."""
    def __init__(self, elem_class, data):
        self.elem_class = elem_class
        self.name = data["property"]
        self.publish = data.get("publish", False)
        self._data_conversion = DataConversion(data.get("data_conversion", "none"))
        self._sum_elements = data.get("sum_elements", False)
        self._limits = self._parse_limits(data)
        self._limits_filter = LimitsFilter(data.get("limits_filter", "outside"))
        self._store_values_type = StoreValuesType(data.get("store_values_type", "all"))
        self._names, self._are_names_regex = self._parse_names(data)
        self._sample_interval = data.get("sample_interval", 1)
        self._window_size = data.get("window_size", 100)
        custom_prop = f"{elem_class}.{self.name}"
        self._custom_metric = CUSTOM_METRICS.get(custom_prop)

        if self._sum_elements and self._store_values_type not in \
                (StoreValuesType.ALL, StoreValuesType.SUM):
            raise InvalidConfiguration(
                "sum_elements requires store_values_types = all or sum"
            )

        if self._is_max() and self._limits is not None:
            raise InvalidConfiguration("limits are not allowed with max types")

    def _is_max(self):
        return self._store_values_type in (
            StoreValuesType.MAX, StoreValuesType.MOVING_AVERAGE_MAX,
        )

    def _is_moving_average(self):
        return self._store_values_type in (
            StoreValuesType.MOVING_AVERAGE, StoreValuesType.MOVING_AVERAGE_MAX,
        )

    @staticmethod
    def _parse_limits(data):
        limits = data.get("limits")
        if limits is None:
            return None

        if not isinstance(limits, list) or len(limits) != 2:
            raise InvalidConfiguration(f"invalid limits format: {limits}")

        return MinMax(limits[0], limits[1])

    @staticmethod
    def _parse_names(data):
        names = data.get("names")
        name_regexes = data.get("name_regexes")

        if names and name_regexes:
            raise InvalidConfiguration(f"names and name_regexes cannot both be set")
        for obj in (names, name_regexes):
            if obj is None:
                continue
            if not isinstance(obj, list) or not isinstance(obj[0], str):
                raise InvalidConfiguration(f"invalid name format: {obj}")

        if names:
            return set(names), False
        if name_regexes:
            return [re.compile(r"{}".format(x)) for x in name_regexes], True

        return None, False

    def _is_inside_limits(self, value):
        """Return True if the value is between min and max, inclusive."""
        return value >= self._limits.min and value <= self._limits.max

    def _is_outside_limits(self, value):
        """Return True if the value is lower than min or greater than max."""
        return value < self._limits.min or value > self._limits.max

    def get_dataset_property_type(self):
        """Return the encoding to use for the dataset backing these values.

        Returns
        -------
        DatasetPropertyType

        """
        if self._limits is not None:
            return DatasetPropertyType.FILTERED
        if self._store_values_type in \
                (StoreValuesType.SUM, StoreValuesType.MAX,
                 StoreValuesType.MOVING_AVERAGE_MAX,
                 StoreValuesType.CHANGE_COUNT):
            return DatasetPropertyType.NUMBER
        return DatasetPropertyType.ELEMENT_PROPERTY

    def get_max_size(self, num_steps):
        """Return the max number of items that could be stored."""
        singles = (
            StoreValuesType.CHANGE_COUNT, StoreValuesType.MAX,
            StoreValuesType.MOVING_AVERAGE_MAX, StoreValuesType.SUM,
        )
        if self._store_values_type in singles:
            return 1
        num_samples = num_steps / self._sample_interval
        return int(num_samples)

    @property
    def data_conversion(self):
        """Return the data_conversion attribute

        Returns
        -------
        DataConversion

        """
        return self._data_conversion

    @property
    def custom_metric(self):
        """Return the custom_metric attribute.

        Returns
        -------
        Metric | None

        """
        return self._custom_metric

    @property
    def limits(self):
        """Return the limits for the export property.

        Returns
        -------
        MinMax

        """
        return self._limits

    def serialize(self):
        """Serialize object to a dictionary."""
        if self._are_names_regex:
            #raise InvalidConfiguration("cannot serialize when names are regex")
            logger.warning("cannot serialize when names are regex")
            names = None
        else:
            names = self._names
        data = {
            "property": self.name,
            "data_conversion": self._data_conversion.value,
            "sample_interval": self._sample_interval,
            "names": names,
            "publish": self.publish,
            "store_values_type": self.store_values_type.value,
            "sum_elements": self.sum_elements,
        }
        if self._limits is not None:
            data["limits"] = [self._limits.min, self._limits.max]
            data["limits_filter"] = self._limits_filter.value
        if self._is_moving_average():
            data["window_size"] = self._window_size

        return data

    def should_store_name(self, name):
        """Return True if name matches the input criteria."""
        if self._names is None:
            return True

        if self._are_names_regex:
            for regex in self._names:
                if regex.search(name):
                    return True
            return False

        return name in self._names

    def should_sample_value(self, step_number):
        """Return True if it's time to read a new value."""
        return step_number % self._sample_interval == 0

    def should_store_value(self, value):
        """Return True if the value meets the input criteria."""
        if self._limits is None:
            return True

        if self._limits_filter == LimitsFilter.OUTSIDE:
            return self._is_outside_limits(value)
        return self._is_inside_limits(value)

    def should_store_timestamp(self):
        """Return True if the timestamp should be stored with the value."""
        return self.limits is not None or \
            self._store_values_type == StoreValuesType.MOVING_AVERAGE

    @property
    def storage_name(self):
        if self._store_values_type in (StoreValuesType.ALL, StoreValuesType.CHANGE_COUNT):
            return self.name
        if self._store_values_type == StoreValuesType.MOVING_AVERAGE:
            return self.name + "Avg"
        if self._store_values_type == StoreValuesType.MOVING_AVERAGE_MAX:
            return self.name + "AvgMax"
        if self._store_values_type == StoreValuesType.MAX:
            return self.name + "Max"
        if self._store_values_type == StoreValuesType.SUM:
            return self.name + "Sum"
        assert False

    @property
    def store_values_type(self):
        """Return the type of storage for this property."""
        return self._store_values_type

    @property
    def sum_elements(self):
        """Return True if the value is the sum of all elements."""
        return self._sum_elements
    @property
    def window_size(self):
        """Return the window size for moving averages.

        Returns
        -------
        int

        """
        return self._window_size


class ExportListReader:
    """Reads export files and provides access to export properties."""
    def __init__(self, filename):
        self._elem_classes = defaultdict(list)
        legacy_files = ("ExportMode-byClass.toml", "ExportMode-byElement.toml")
        if os.path.basename(filename) in legacy_files:
            parser = self._parse_legacy_file
        else:
            parser = self._parse_file

        for elem_class, data in parser(filename):
            self._elem_classes[elem_class].append(ExportListProperty(
                elem_class, data
            ))

    @staticmethod
    def _parse_file(filename):
        data = load_data(filename)
        for elem_class, prop_info in data.items():
            if isinstance(prop_info, list):
                for prop in prop_info:
                    yield elem_class, prop
            else:
                assert isinstance(prop_info, dict)
                for prop, values in prop_info.items():
                    new_data = {"property": prop, **values}
                    yield elem_class, new_data

    @staticmethod
    def _parse_legacy_file(filename):
        reader = pyExportReader(filename)
        publications = {tuple(x.split()) for x in reader.publicationList}
        for elem_class, props in reader.pyControllers.items():
            for prop in props:
                publish = (elem_class, prop) in publications
                yield elem_class, {"property": prop, "publish": publish}

    def append_property(self, elem_class, prop_data):
        self._elem_classes[elem_class].append(ExportListProperty(elem_class, prop_data))

    def get_element_properties(self, elem_class, prop):
        if elem_class not in self._elem_classes:
            raise InvalidParameter(f"{elem_class} is not stored")
        return [x for x in self._elem_classes[elem_class] if x.name == prop]

    def iter_export_properties(self, elem_class=None):
        """Returns a generator over the ExportListProperty objects.

        Yields
        ------
        ExportListProperty

        """
        if elem_class is None:
            for props in self._elem_classes.values():
                for prop in props:
                    yield prop
        elif elem_class not in self._elem_classes:
            raise InvalidParameter(f"{elem_class} is not stored")
        else:
            for prop in self._elem_classes[elem_class]:
                yield prop

    def list_element_classes(self):
        return sorted(list(self._elem_classes.keys()))

    def list_element_properties(self, elem_class):
        if elem_class not in self._elem_classes:
            return []
        return self._elem_classes[elem_class]

    def list_element_property_names(self, elem_class):
        return sorted({x.name for x in self._elem_classes[elem_class]})

    # This name needs to match the interface defined in pyExportReader.
    @property
    def publicationList(self):
        """Return the properties to be published to HELICS.

        Returns
        -------
        list
            Format: ["ElementClass Property"]

        """
        return [
            f"{x.elem_class} {x.name}" for x in self.iter_export_properties()
            if x.publish
        ]

    def serialize(self):
        """Serialize object to a dictionary."""
        data = defaultdict(list)
        for elem_class, props in self._elem_classes.items():
            for prop in props:
                data[elem_class].append(prop.serialize())

        return data
