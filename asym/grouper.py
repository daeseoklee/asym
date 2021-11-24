from typing import Tuple, List, Dict, Any, Callable, Union
from abc import ABCMeta, abstractmethod

import torch 
from torch import Tensor
from torch.nn.functional import pad

from asym.shape_signature import *
from asym.data import TensorData

class DataListGrouper(metaclass=ABCMeta):
    def __init__(self):
        pass 

    @abstractmethod
    def get_data_partition(self, data_list:List[TensorData], shapesig_data:ShapeSignatureData) -> List[List[int]]:
        pass

class UniGrouper(DataListGrouper):
    def __init__(self):
        super().__init__()
    def get_data_partition(self, data_list:List[TensorData], shapesig_data:ShapeSignatureData) -> List[List[int]]:
        return [list(range(len(data_list)))]
    
class PredefinedGrouper(DataListGrouper):
    def __init__(self, partition:List[List[int]]):
        super().__init__()
        self.partition = partition
        self.length = sum([len(part) for part in partition])
    def get_data_partition(self, data_list:List[TensorData], shapesig_data:ShapeSignatureData):
        assert len(data_list) == self.length
        return self.partition
    
class PredefinedConsecutiveGrouper(PredefinedGrouper):
    def __init__(self, nums:List[int]):
        partition = [] 
        sofar = 0
        for num in nums:
            partition.append(list(range(sofar, sofar + num)))
            sofar += num
        super().__init__(partition)
    

class LengthBasedGrouper(DataListGrouper, metaclass=ABCMeta):
    def __init__(self, ldim_label:str):
        super().__init__()
        self.ldim_label = ldim_label
    def get_data_partition(self, data_list:List[TensorData], shapesig_data:ShapeSignatureData) -> List[List[int]]:
        keys, idx = shapesig_data.prestack_ldim_map[self.ldim_label]
        lengths = [data.get(keys).value.shape[idx] for data in data_list]
        return self.get_partition_from_lengths(lengths)
    @abstractmethod 
    def get_partition_from_lengths(self, lengths:List[int]):
        pass
    
class LengthThresholdGrouper(LengthBasedGrouper):
    def __init__(self, ldim_label, thresholds):
        super().__init__(ldim_label)
        self.thresholds = sorted(thresholds)
        assert len(self.thresholds) >= 1
        for i in range(len(self.thresholds) - 1):
            assert self.thresholds[i] < self.thresholds[i + 1]
    def get_subgroup_idx(self, num):
        def _search(min_idx=0, max_idx=len(self.thresholds)):
            if min_idx == max_idx:
                return min_idx
            assert min_idx < max_idx 
            middle_idx = (min_idx + max_idx) // 2
            if num >= self.thresholds[middle_idx]:
                return _search(middle_idx + 1, max_idx)
            else:
                return _search(min_idx, middle_idx)
        return _search()
    def get_partition_from_lengths(self, lengths:List[int]):
        partition = [[] for _ in range(len(self.thresholds) + 1)]
        for i, num in enumerate(lengths):
            partition[self.get_subgroup_idx(num)].append(i)
        partition = [part for part in partition if len(part) > 0]
        return partition