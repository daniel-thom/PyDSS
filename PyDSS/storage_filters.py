
import abc
import copy
import logging

import numpy as np

from PyDSS.common import StoreValuesType
from PyDSS.utils.simulation_utils import CircularBufferHelper
from PyDSS.value_storage import ValueContainer

logger = logging.getLogger(__name__)


class StorageFilterBase(abc.ABC):
    """Base class for storage containers.
    Subclasses can perform custom filtering based on StoreValuesType.

    """
    def __init__(self, hdf_store, path, prop, num_steps, max_chunk_bytes, values, elem_names):
        self._prop = prop
        self._container = self.make_container(
            hdf_store,
            path,
            prop,
            num_steps,
            max_chunk_bytes,
            values,
            elem_names,
        )
        logger.debug("Created %s path=%s", self.__class__.__name__, path)

    @abc.abstractmethod
    def append_values(self, values, time_step):
        """Store a new set of values for each element."""

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
    def make_container(hdf_store, path, prop, num_steps, max_chunk_bytes, values, elem_names):
        """Return an instance of ValueContainer for storing values."""
        container = ValueContainer(
            values,
            hdf_store,
            path,
            prop.get_max_size(num_steps),
            elem_names,
            prop.get_dataset_property_type(),
            max_chunk_bytes=max_chunk_bytes,
            store_time_step=prop.should_store_time_step(),
        )
        logger.debug("Created storage container path=%s", path)
        return container


class StorageAll(StorageFilterBase):
    """Store values at every time point, optionally filtered."""

    def append_values(self, values, time_step):
        if self._prop.limits:
            for i, value in enumerate(values):
                if self._prop.should_store_value(value.value):
                    self._container.append_by_time_step(value, time_step, i)
        else:
            self._container.append(values)


"""
class StorageChangeCount(StorageFilterBase):
    def __init__(self, *args):
        super().__init__(*args)
        self._last_value = None
        self._change_count = (None, 0)

    def append_values(self, values, time_step):
        assert False
"""


class StorageMax(StorageFilterBase):
    """Stores the max value across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._max = None

    def append_values(self, values, time_step):
        self._handle_values(values)

    def close(self):
        self._container.append(self._max)
        self._container.flush_data()

    def _handle_values(self, values):
        if self._max is None:
            self._max = values
        else:
            for i, new_val in enumerate(values):
                cur_val = self._max[i]
                if (np.isnan(cur_val.value) and not np.isnan(new_val.value)) or \
                        new_val > cur_val:
                    self._max[i] = new_val


class StorageMovingAverage(StorageFilterBase):
    """Stores a moving average across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._bufs = None

    def append_values(self, values, time_step):
        # Store every value in the circular buffer. Apply limits to the
        # moving average.
        if self._bufs is None:
            self._bufs = [CircularBufferHelper(self._prop) for _ in range(len(values))]
        averages = [0] * len(values)
        for i, val in enumerate(values):
            buf = self._bufs[i]
            buf.append(val.value)
            averages[i] = buf.average()
        logger.debug("averages=%s", averages)

        if self._prop.limits:
            for i, avg in enumerate(averages):
                if self._prop.should_store_value(avg):
                    # TODO: perf issue?
                    new_value = copy.deepcopy(values[i])
                    new_value.set_element_property(self._prop.storage_name)
                    new_value.set_value(avg)
                    logger.debug("append filtered average %s time_step=%s, elem=%i",
                                 new_value.value, time_step, i)
                    self._container.append_by_time_step(new_value, time_step, i)
        else:
            new_values = []
            for i, avg in enumerate(averages):
                new_value = copy.deepcopy(values[i])
                new_value.set_element_property(self._prop.storage_name)
                new_value.set_value(avg)
                new_values.append(new_value)
            self._container.append(new_values)


class StorageMovingAverageMax(StorageMax):
    """Stores the max value of a moving average across time points."""
    def __init__(self, *args):
        super().__init__(*args)
        self._bufs = None

    def append_values(self, values, time_step):
        if self._bufs is None:
            self._bufs = [CircularBufferHelper(self._prop) for _ in range(len(values))]
        averages = [0] * len(values)
        for i, val in enumerate(values):
            buf = self._bufs[i]
            buf.append(val.value)
            new_value = copy.deepcopy(val)
            new_value.set_element_property(self._prop.storage_name)
            new_value.set_value(buf.average())
            averages[i] = new_value
        self._handle_values(averages)

        #self._buf.append(value.value)
        #avg = self._buf.average()
        #logger.info("append avg value value=%s avg=%s", value.value, avg)
        #new_value = copy.deepcopy(value)
        #new_value.set_element_property(self._prop.storage_name)
        #new_value.set_value(avg)
        #self._handle_values(new_values)


class StorageSum(StorageFilterBase):
    """Keeps a running sum of all values and records the total."""
    def __init__(self, *args):
        super().__init__(*args)
        self._sum = None

    def append_values(self, values, _time_step):
        if self._sum is None:
            self._sum = values
        else:
            for i, val in enumerate(values):
                self._sum[i] += val

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
