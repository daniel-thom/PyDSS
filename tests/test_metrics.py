
from collections import namedtuple
import os
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import pytest

from PyDSS.common import LimitsFilter
from PyDSS.dataset_buffer import DatasetBuffer
from PyDSS.export_list_reader import ExportListProperty
from PyDSS.metrics import MultiValueTypeMetrics
from PyDSS.value_storage import ValueByNumber, ValueByList
from PyDSS.utils.utils import load_data


OPTIONS = load_data("PyDSS/defaults/simulation.toml")
STORE_FILENAME = os.path.join(tempfile.gettempdir(), "store.h5")
FLOATS = (1.0, 2.0, 3.0, 4.0, 5.0)
COMPLEX_NUMS = (
    complex(1, 2), complex(3, 4), complex(5, 6), complex(7, 8),
    complex(9, 10),
)
LIST_COMPLEX_NUMS = (
    [complex(1, 2), complex(3, 4)],
    [complex(5, 6), complex(7, 8)],
    [complex(9, 10), complex(11, 12)],
    [complex(13, 14), complex(15, 16)],
    [complex(17, 18), complex(19, 20)],
)


FakeObj = namedtuple("FakeObj", "FullName, Name")


@pytest.fixture
def cleanup():
    if os.path.exists(STORE_FILENAME):
        os.remove(STORE_FILENAME)
    yield
    if os.path.exists(STORE_FILENAME):
        os.remove(STORE_FILENAME)


class FakeMetric(MultiValueTypeMetrics):

    def __init__(self, prop, dss_obj, options, values):
        super().__init__(prop, dss_obj, options)
        self._index = 0
        self._values = values

    def _get_value(self, timestamp):
        obj = self._dss_objs[0]
        prop = next(iter(self._properties.values()))
        if isinstance(self._values[self._index], list):
            val = ValueByList(obj.FullName, prop.name, self._values[self._index], ["", ""])
        else:
            val = ValueByNumber(obj.FullName, prop.name, self._values[self._index])
        self._index += 1
        if self._index == len(self._values):
            self._index = 0
        return val


def test_metrics_store_all(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "all",
    }
    values = FLOATS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/Property"]
        assert dataset.attrs["length"] == len(values)
        assert dataset.attrs["type"] == "elem_prop"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        assert len(series) == len(values)
        for val1, val2 in zip(series.values, values):
            assert val1 == val2
        assert metric.max_num_bytes() == len(values) * 8


def test_metrics_store_all_complex_abs(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "all",
        "data_conversion": "abs",
    }
    values = COMPLEX_NUMS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/Property"]
        assert dataset.attrs["length"] == len(values)
        assert dataset.attrs["type"] == "elem_prop"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        assert len(series) == len(values)
        for val1, val2 in zip(series.values, values):
            assert isinstance(val1, float)
            assert val1 == abs(val2)


def test_metrics_store_all_complex_sum(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "all",
        "data_conversion": "sum",
    }
    values = LIST_COMPLEX_NUMS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/Property"]
        assert dataset.attrs["length"] == len(values)
        assert dataset.attrs["type"] == "elem_prop"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        assert len(series) == len(values)
        for val1, val2 in zip(series.values, values):
            assert isinstance(val1, complex)
            assert val1 == sum(val2)


def test_metrics_store_all_complex_abs_sum(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "all",
        "data_conversion": "abs_sum",
    }
    values = LIST_COMPLEX_NUMS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/Property"]
        assert dataset.attrs["length"] == len(values)
        assert dataset.attrs["type"] == "elem_prop"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        assert len(series) == len(values)
        for val1, val2 in zip(series.values, values):
            assert isinstance(val1, float)
            assert val1 == abs(sum(val2))


def test_metrics_store_all_filtered(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "all",
        "limits": [1.0, 3.0],
        "limits_filter": LimitsFilter.OUTSIDE,
    }
    values = FLOATS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/Property"]
        assert dataset.attrs["length"] == 2
        assert dataset.attrs["type"] == "filtered"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        series = df.iloc[:, 0]
        assert series.values[0] == 4.0
        assert series.values[1] == 5.0


def test_metrics_store_moving_average_and_max(cleanup):
    window_size = 10
    values = [float(i) for i in range(100)]
    data1 = {
        "property": "Property",
        "store_values_type": "max",
    }
    data2 = {
        "property": "Property",
        "store_values_type": "moving_average",
        "window_size": window_size,
    }
    prop1 = ExportListProperty("Fake", data1)
    prop2 = ExportListProperty("Fake", data2)
    obj = FakeObj("Fake.name", "name")
    metric = FakeMetric(prop1, obj, OPTIONS, values)
    metric.add_property(prop2)

    base_df = pd.DataFrame(values)
    base_series = base_df.iloc[:, 0]
    base_rm = base_series.rolling(window_size).mean()

    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/PropertyAvg"]
        assert dataset.attrs["length"] == len(values)
        assert dataset.attrs["type"] == "elem_prop"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        for val1, val2 in zip(series.values, base_rm.values):
            if np.isnan(val1):
                assert np.isnan(val2)
            else:
                assert val1 == val2

        dataset2 = hdf_store[f"Fake/Elements/{obj.FullName}/PropertyMax"]
        assert dataset2.attrs["length"] == 1
        assert dataset2.attrs["type"] == "number"
        assert dataset2[0] == 99.0


def test_metrics_store_moving_average_with_limits(cleanup):
    limits = [1.0, 50.0]
    window_size = 10
    data = {
        "property": "Property",
        "store_values_type": "moving_average",
        "window_size": window_size,
        "limits": limits,
        "limits_filter": LimitsFilter.OUTSIDE,
    }
    values = [float(x) for x in range(1, 101)]
    #values = [float(x) for _ in range(10) for x in range(10)]
    expected_values = [x for x in values if x < limits[0] or x > limits[1]]
    base_df = pd.DataFrame(values)
    base_series = base_df.iloc[:, 0]
    base_rm = base_series.rolling(window_size).mean()
    expected = [x for x in base_rm.values if x < limits[0] or x > limits[1]]
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/PropertyAvg"]
        assert dataset.attrs["length"] == len(expected)
        assert dataset.attrs["type"] == "filtered"
        df = DatasetBuffer.to_dataframe(dataset)
        assert isinstance(df, pd.DataFrame)
        series = df.iloc[:, 0]
        for val1, val2 in zip(series.values, expected):
            assert val1 == val2


def test_metrics_store_moving_average_max(cleanup):
    window_size = 10
    data = {
        "property": "Property",
        "store_values_type": "moving_average_max",
        "window_size": window_size,
    }
    values = [float(i) for i in range(100)]
    base_df = pd.DataFrame(values)
    base_series = base_df.iloc[:, 0]
    base_rm = base_series.rolling(window_size).mean()
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/PropertyAvgMax"]
        assert dataset.attrs["length"] == 1
        assert dataset.attrs["type"] == "number"
        val = dataset[0]
        assert val == base_rm.max()
        assert metric.max_num_bytes() == 8


def test_metrics_store_sum(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "sum",
    }
    values = FLOATS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/PropertySum"]
        assert dataset.attrs["length"] == 1
        assert dataset.attrs["type"] == "number"
        assert dataset[:][0] == sum(values)
        assert metric.max_num_bytes() == 8


def test_metrics_store_max(cleanup):
    data = {
        "property": "Property",
        "store_values_type": "max",
    }
    values = FLOATS
    obj = FakeObj("Fake.name", "name")
    prop = ExportListProperty("Fake", data)
    metric = FakeMetric(prop, obj, OPTIONS, values)
    with h5py.File(STORE_FILENAME, mode="w", driver="core") as hdf_store:
        metric.initialize_data_store(hdf_store, "", len(values))
        for _ in range(len(values)):
            metric.append_values(time.time())
        metric.close()

        dataset = hdf_store[f"Fake/Elements/{obj.FullName}/PropertyMax"]
        assert dataset.attrs["length"] == 1
        assert dataset.attrs["type"] == "number"
        assert dataset[:][0] == max(values)
        assert metric.max_num_bytes() == 8
