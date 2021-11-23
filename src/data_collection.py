from typing import Tuple, List, Dict, Any, Callable, Union
from abc import ABCMeta, abstractmethod

import torch 
from torch import Tensor
from torch.nn.functional import pad

from shape_signature import *

from data import TensorData
from grouper import DataListGrouper
from shape_signature import ShapeSignature, ShapeSignatureData, PreShapeSignature, PreShapeSignatureData
#from shape_match import get_output_shape
from padding import PadderData, Padder, ZeroPadder, merge_tensors
from annotated_module import AnnotatedModule



class DataCollection:
    def __init__(self, shape_annot:Union[ShapeSignatureData, Dict[str, Union[str, Any]]], data_list:List[TensorData]=None, data_groups:List[TensorData]=None, length_info:Dict[str, List[List[int]]]=None, data_partition:List[Tuple[int, int]]=None,validate=False):
        """ 
        There are two ways of initializing DataCollection:
            1. providing a data_list 
            2. providing 
                -data_groups 
                -length_info
                -data_partition (optional)
        It is always either in "grouped" or "grouped" form. Being grouped means the data list is partitioned into several parts and each part form a singe TensorData object after paddings
        """
        if type(shape_annot) == ShapeSignatureData:
            self.shapesig_data = shape_annot
        else:
            self.shapesig_data = ShapeSignatureData.parse(shape_annot)
        if data_list is not None:
            if data_groups is not None or length_info is not None or data_partition is not None:
                raise Exception('either data_list or all of [data_groups, length_info, data_partition] should be provided')
            self.data_list = data_list
            self.data_groups = None
            self.length_info = None
            self.data_partition = None
            self.is_grouped = False
        if data_groups is not None:
            if data_list is not None or length_info is None:
                raise Exception('either data_list or all of [data_groups, length_info] should be provided')
            if data_partition is None:
                data_partition = DataCollection.get_default_data_partition(data_groups, self.shapesig_data) 
            self.data_list = None
            self.data_groups = data_groups
            self.length_info = length_info
            self.data_partition = data_partition
            self.is_grouped = True
        
        if validate:
            self.validate()
            
    def __len__(self):
        if self.is_grouped:
            keys, idx = self.shapesig_data.bdim_loc
            group_lens = [group.get(keys).value.shape[idx] for group in self.data_groups]
            return sum(group_lens)
        return len(self.data_list)
    
    @classmethod
    def get_default_data_partition(cls, data_groups:List[TensorData], shapesig_data:ShapeSignatureData) -> List[List[int]]:
        def get_any_keys(shapesig_data:ShapeSignatureData, acc=[]):
            if shapesig_data.is_leaf:
                return acc 
            key = shapesig_data.keys()[0]
            subdata = shapesig_data[key]
            return get_any_keys(subdata, acc=acc+[key])
        
        keys, loc = shapesig_data.bdim_loc 
        batch_sizes = [group.get(keys).value.shape[loc] for group in data_groups]
        partition = []
        sofar = 0
        for batch_size in batch_sizes:
            partition.append(list(range(sofar, sofar + batch_size)))
            sofar += batch_size
        return partition
    
    def validate(self):
        def _validate_data_list_shape():
            pass
        def _validate_data_group_shape():
            pass
        if not self.is_grouped:
            _validate_data_list_shape()
        else:
            _validate_data_group_shape()
            
    def get_length_info(self, partition:List[List[int]]) -> Dict[str, List[List[int]]]: 
        if self.is_grouped:
            return self.length_info
        assert self.data_list is not None and self.shapesig_data is not None
        
        length_info = {}
        for label, (keys, idx) in self.shapesig_data.prestack_ldim_map.items():
            length_info[label] = [[self.data_list[i].get(keys).value.shape[idx] for i in part] for part in partition]
        return length_info
    
    def get_group_length_info(self, i:int) -> Dict[str, List[int]]:
        assert self.is_grouped
        return {key: val[i] for key, val in self.length_info.items()}
        
    
    def get_data_groups(self, partition:List[List[int]], padder_data:PadderData) -> List[TensorData]:
        if self.is_grouped:
            return self.data_groups
        assert self.data_list is not None
        
        def _get_data_group(data_sublist:List[TensorData], shapesig_data:ShapeSignatureData, padder_data:PadderData) -> TensorData:
            first_data = data_sublist[0]
            if not first_data.is_leaf:
                d = {key: _get_data_group([data[key] for data in data_sublist], shapesig_data[key], padder_data[key]) for key in first_data.keys()}
                return TensorData.from_data_dict(d)

            assert padder_data.is_leaf
            padder = padder_data.value
            
            assert hasattr(padder, 'pad')
            assert shapesig_data.is_leaf
            shapesig = shapesig_data.value
            return TensorData(merge_tensors([data.value for data in data_sublist], shapesig, padder))
        
        return [_get_data_group([self.data_list[i] for i in part], self.shapesig_data, padder_data) for part in partition]
            
    
    def get_data_list(self) -> List[TensorData]:
        if not self.is_grouped:
            assert self.data_list is not None
            return self.data_list
        assert self.data_groups is not None
        if self.data_partition is None:
            data_partition = DataCollection.get_default_data_partition(self.data_groups, self.shapesig_data)
        else:
            data_partition = self.data_partition
            
        data_list = [None for _ in range(len(self))]
        
        def get_sliced_tensor(t:Tensor, prestack_shapesig:ShapeSignature, length_hint:Dict[str, int]):
            for idx, label in prestack_shapesig.iter_ldim_idx_label():
                length = length_hint[label]
                t = t.index_select(idx, torch.arange(length, device=t.device))
            return t 
        
        def get_data_sublist(group:TensorData, shapesig_data:ShapeSignatureData, length_hints:List[Dict[str, int]]) -> List[TensorData]:
            if shapesig_data.is_leaf:
                shapesig:ShapeSignature = shapesig_data.value 
                assert shapesig.bdim_idx != -1 
                assert group.is_leaf 
                bdim_idx = shapesig.bdim_idx 
                prestack_shapesig = shapesig.bdim_removed()
                t:Tensor = group.value
                data_sublist = [TensorData(get_sliced_tensor(t.index_select(bdim_idx, torch.tensor(i, device=t.device)).squeeze(bdim_idx), prestack_shapesig, length_hints[i])) for i in range(len(length_hints))]
                return data_sublist
            sublist_dict = {}
            for key in shapesig_data.keys():
                sublist_dict[key] = get_data_sublist(group[key], shapesig_data[key], length_hints)
            return [TensorData.from_data_dict({key: sublist_dict[key][i] for key in shapesig_data.keys()}) for i in range(len(length_hints))]
            
        for i, (part, group) in enumerate(zip(data_partition, self.data_groups)):
            length_hints = [{key: val[i][j] for key, val in self.length_info.items()} for j in range(len(part))]
            data_sublist = get_data_sublist(group, self.shapesig_data, length_hints)
            for idx, data in zip(part, data_sublist):
                data_list[idx] = data
        return data_list
            
        
    def group(self, grouper:DataListGrouper, padding:Union[Padder, Dict[str, Any]]=None, forget_order=False):
        if self.is_grouped:
            raise Exception('Already grouped, call ungroup() first')
        padder_data = PadderData(padding, template=self.shapesig_data.get_template())
        data_partition = grouper.get_data_partition(self.data_list, self.shapesig_data)
        self.length_info = self.get_length_info(data_partition)
        self.data_groups = self.get_data_groups(data_partition, padder_data)
        if forget_order:
            self.data_partition = None
        else:
            self.data_partition = data_partition
        self.data_list = None
        self.is_grouped = True
        
    def ungroup(self):
        if not self.is_grouped:
            raise Exception('Already ungrouped, call group() first')
        assert self.data_groups is not None and self.length_info is not None
        self.data_list = self.get_data_list()
        self.data_groups = None
        self.length_info = None
        self.data_partition = None
        self.is_grouped = False

    def get_grand_batch(self):
        pass
    
    def get_mask_data(self, i:int, key_conv=None) -> TensorData:
        group = self.data_groups[i]
        length_info = self.get_group_length_info(i)
        def _get_mask_d(t:Union[Tensor, Dict[str, Any]], shapesig:Union[ShapeSignature, Dict[str, Any]]):
            if type(shapesig) == dict:
                assert type(t) == dict
                return {key: _get_mask_d(t[key], shapesig[key]) for key in shapesig}
            assert type(shapesig) == ShapeSignature
            assert type(t) == Tensor 
            bdim_idx = shapesig.bdim_idx 
            ldim_label_map = {idx: label for idx, label in shapesig.iter_ldim_idx_label()}
            masks = [] 
            mask_shape = t.shape[:bdim_idx] + (1,) + t.shape[bdim_idx:]
            for i in range(t.shape[bdim_idx]):
                axis_tensors = [torch.tensor([k < length_info[ldim_label_map[dim_idx]][i] if dim_idx in ldim_label_map else dim_size for k in range(dim_size)], device=t.device) for dim_idx, dim_size in enumerate(t.shape) if dim_idx != bdim_idx]
                mask = torch.cartesian_prod(*axis_tensors).all(dim=-1).reshape(mask_shape)
                masks.append(mask)
            return torch.cat(masks, dim=bdim_idx)
        
        mask_data = TensorData(_get_mask_d(group.value, self.shapesig_data.value))
        if key_conv is not None:
            mask_data = mask_data.keys_converted(key_conv=key_conv)
        return mask_data
                

    def apply(self, f:AnnotatedModule, require_mask=True, input_key_conv=None, output_key_conv=None) -> 'DataCollection':
        """
        """
        if not self.is_grouped:
            raise Exception('Currently can apply AnnotatedModule only when the DataCollection is "grouped"')
        
        output_shapesig_data = type(f).get_output_shapesig_data(self.shapesig_data, input_key_conv=input_key_conv, output_key_conv=output_key_conv)
        
        key_converted_data_groups = [group.keys_converted(input_key_conv) for group in self.data_groups]
        if require_mask:
            new_data_groups = [TensorData(f(group.value, self.get_mask_data(i, key_conv=input_key_conv).value)) for i, group in enumerate(key_converted_data_groups)]
        else:
            new_data_groups = [TensorData(f(group.value)) for group in key_converted_data_groups]
        
        if output_key_conv is not None:
            new_data_groups = [group.keys_converted(output_key_conv) for group in new_data_groups]
                
        new_length_info = {label: info for label, info in self.length_info.items() if label in output_shapesig_data.ldim_map}

        return DataCollection(output_shapesig_data, data_groups=new_data_groups, length_info=new_length_info, data_partition=self.data_partition)
    
    def __getitem__(self, key:str) -> 'DataCollection':
        if self.shapesig_data.is_leaf:
            raise Exception('The data consists of tensors(not dicts)')
        if not key in self.shapesig_data.keys():
            raise Exception(f'No key {key}')
        shapesig_data:ShapeSignatureData = self.shapesig_data[key]
        if self.is_grouped:
            data_groups = [group[key] for group in self.data_groups]
            length_info = {key: val for key, val in self.length_info.items() if key in shapesig_data.ldim_map}
            return DataCollection(shapesig_data, data_groups=data_groups, length_info=length_info, data_partition=self.data_partition)
        else:
            data_list = [data[key] for data in self.data_list]
            return DataCollection(shapesig_data, data_list=data_list)
    
    @classmethod 
    def combine(cls, dc_dict:Union['DataCollection', Dict[str, Any]]) -> 'DataCollection':
        pass

