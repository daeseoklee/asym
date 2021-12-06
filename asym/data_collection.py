from typing import Tuple, List, Dict, Any, Callable, Union
from abc import ABCMeta, abstractmethod
from time import time

import torch 
from torch import Tensor
from torch.nn.functional import pad

from asym.shape_signature import *

from asym.data import TensorData
from asym.grouper import DataListGrouper
from asym.shape_signature import ShapeSignature, ShapeSignatureData, PreShapeSignature, PreShapeSignatureData
from asym.padding import PadderData, Padder
from asym.annotated_module import AnnotatedModule


class IncompatibleDataCollectionError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class DataCollection:
    def __init__(self, shape_annot:Union[ShapeSignatureData, Dict[str, Union[str, Any]]], data_list:List[TensorData]=None, data_groups:List[TensorData]=None, length_info:Dict[str, List[List[int]]]=None, data_partition:List[Tuple[int, int]]=None, padding:Union[PadderData, Padder, Dict[str, Any]]=None, validate=False):
        """ 
        There are two ways of initializing DataCollection:
            1. providing a data_list 
            2. providing 
                -data_groups 
                -length_info
                -data_partition (optional)
        It is always either in "grouped" or "ungrouped" form. Being grouped means the data list is partitioned into several parts and each part form a singe TensorData object after paddings
        """
        if type(shape_annot) == ShapeSignatureData:
            self.shapesig_data = shape_annot
        else:
            self.shapesig_data = ShapeSignatureData.parse(shape_annot)

        if padding is not None:
            if type(padding) == PadderData:
                self.preset_padder_data = PadderData(padding.value, template=self.shapesig_data.get_template())
            else:
                self.preset_padder_data = PadderData(padding, template=self.shapesig_data.get_template())
        else:
            self.preset_padder_data = None
            
        if data_list is not None:
            if data_groups is not None or length_info is not None or data_partition is not None:
                raise Exception('either data_list or all of [data_groups, length_info, data_partition] should be provided')
            self.data_list = [data if type(data) == TensorData else TensorData(data) for data in data_list]
            self.data_groups = None
            self.length_info = None
            self.data_partition = None
            self.is_grouped = False
        elif data_groups is not None:
            if data_list is not None or length_info is None:
                raise Exception('either data_list or all of [data_groups, length_info] should be provided')
            if type(data_groups) != list:
                raise Exception('data_groups should be a list')
            if len(data_groups) == 0:
                raise Exception('data_groups should be a non-empty list')
            self.data_list = None
            if type(data_groups[0]) == list:
                pass
            else:
                self.data_groups = [group if type(group) == TensorData else TensorData(group) for group in data_groups]
            if data_partition is None:
                data_partition = DataCollection.get_default_data_partition(data_groups, self.shapesig_data) 
            self.length_info = length_info
            self.data_partition = data_partition
            self.is_grouped = True
        else:
            raise Exception('either data_list or all of [data_groups, length_info] should be provided')

        if validate:
            self.validate()

            
    def get_group_hash(self):
        if self.is_grouped:
            assert self.data_partition is not None 
            assert type(self.data_partition) == list
            assert type(self.data_partition[0]) == list
            assert type(self.data_partition[0][0]) == int 
            return hash(tuple([hash(tuple(part)) for part in self.data_partition]))
        assert self.data_list is not None
        return len(self.data_list)

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
        
    
    def get_data_group(self, data_sublist:List[TensorData], shapesig_data:ShapeSignatureData, padder_data:PadderData) -> TensorData:
        first_data = data_sublist[0]
        if not first_data.is_leaf:
            d = {key: self.get_data_group([data[key] for data in data_sublist], shapesig_data[key], padder_data[key]) for key in first_data.keys()}
            return TensorData.from_data_dict(d)

        assert padder_data.is_leaf
        padder = padder_data.value
        
        assert hasattr(padder, 'pad')
        assert shapesig_data.is_leaf
        shapesig = shapesig_data.value
        return TensorData(padder.merge_tensors([data.value for data in data_sublist], shapesig))
    
    
    def get_data_groups(self, partition:List[List[int]], padder_data:PadderData) -> List[TensorData]:
        if self.is_grouped:
            return self.data_groups
        assert self.data_list is not None
        
        return [self.get_data_group([self.data_list[i] for i in part], self.shapesig_data, padder_data) for part in partition]
            
    
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
        if self.preset_padder_data is not None:
            if padding is not None:
                raise Exception('There are pre-set padders')
            padder_data = self.preset_padder_data
        else:
            padder_data = PadderData(padding, template=self.shapesig_data.get_template())
        data_partition = grouper.get_data_partition(self.data_list, self.shapesig_data)
        self.length_info = self.get_length_info(data_partition)
        self.data_groups = self.get_data_groups(data_partition, padder_data)
        if forget_order:
            self.data_partition = self.get_default_data_partition(self.data_groups, self.shapesig_data)
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

    def regroup(self, grouper:DataListGrouper, padding:Union[Padder, Dict[str, Any]]=None, forget_order=False):
        self.ungroup()
        self.group(grouper, padding=padding, forget_order=forget_order)
        
    def _get_new_shapesig(self, shapesig: ShapeSignature, new_bdim_idx:int, new_ldim_label:str):
        if new_bdim_idx >= 0:
            if new_bdim_idx > len(shapesig):
                raise Exception(f'new_bdim_idx {new_bdim_idx} is too big compared to {shapesig}')
            _new_bdim_idx = 0
        else:
            if new_bdim_idx < - len(shapesig) - 1:
                raise Exception(f'new_bdim_idx {new_bdim_idx} is too small compared to {shapesig}')
            _new_bdim_idx = new_bdim_idx + len(shapesig) + 1
        l = []
        for dim in shapesig.list:
            if type(dim) == BDim:
                l.append(LDim(new_ldim_label))
            elif type(dim) == LDim and dim.label == new_ldim_label:
                raise Exception(f'The LDim label {dim.label} already exists in {shapesig}')
            else:
                l.append(dim)
        l.insert(_new_bdim_idx, BDim())
        return ShapeSignature(l)
        
    def groups_as_data(self, bdim_idx:int, new_ldim_label:str, padding=None):
        if not self.is_grouped:
            raise Exception('"groups_as_data()" can be called only when the DataCollection is grouped')
        data_list = self.data_groups
        
        shapesig_conversion = lambda shapesig: self._get_new_shapesig(shapesig, bdim_idx, new_ldim_label)
        shapesig_data = ShapeSignatureData.map(shapesig_conversion, self.shapesig_data)
        
        if padding is None: #We assume that after groups_as_data, same preset_padder_data works
            padding = self.preset_padder_data
        
        return DataCollection(shapesig_data, data_list=data_list, padding=padding)
    
    def masks_as_data(self, bdim_idx:int, new_ldim_label:str, padding=None):
        if not self.is_grouped:
            raise Exception('"masks_as_data()" can be called only when the DataCollection is grouped')
        data_list = [self.get_mask(i, mask_hint='copy') for i in range(len(self.data_groups))]

        shapesig_conversion = lambda shapesig: self._get_new_shapesig(shapesig, bdim_idx, new_ldim_label)
        shapesig_data = ShapeSignatureData.map(shapesig_conversion, self.shapesig_data)
        
        return DataCollection(shapesig_data, data_list=data_list, padding=padding)

    def get_grand_batch(self):
        pass
    
    def get_mask(self, i:int, key_conv=None, mask_hint:Union[str, List[str]]='copy') -> Union[Tensor, Dict[str, Any]]:
        length_info = self.get_group_length_info(i)
        group = self.data_groups[i]
        
        if mask_hint == 'copy':
            def _get_mask(t:Tensor, shapesig:ShapeSignature):

                bdim_idx = shapesig.bdim_idx 
                ldim_label_map = {idx: label for idx, label in shapesig.iter_ldim_idx_label()}
                if len(ldim_label_map) == 0:
                    return torch.ones_like(t, dtype=bool)
                inst_mask_shape = t.shape[:bdim_idx] + (1,) + t.shape[bdim_idx+1:]
                
                def get_shape_at(dim_idx):
                    return (1,) * dim_idx + (t.shape[dim_idx],) + (1,) * (len(t.shape) - dim_idx - 1)
                
                def _get_mask_at(inst_idx):
                    masks = [(torch.arange(t.shape[ldim_idx], device=t.device) < length_info[ldim_label][inst_idx]).reshape(get_shape_at(ldim_idx)).broadcast_to(inst_mask_shape) for ldim_idx, ldim_label in ldim_label_map.items()]
                    return torch.any(torch.stack(masks, dim=0), dim=0)
                
                masks = [_get_mask_at(inst_idx) for inst_idx in range(t.shape[bdim_idx])]
                return torch.cat(masks, dim=bdim_idx) 
            
            mask_data = TensorData.map2(_get_mask, group, self.shapesig_data)
            if key_conv is not None:
                mask_data = mask_data.keys_converted(key_conv=key_conv)
            return mask_data.value
        
        elif type(mask_hint) == list:
            device = group.get_any_leaf_value().device
            if len(mask_hint) == 0:
                raise Exception(f'mask_hint should be a non-empty list of strings')
            for ldim_label in mask_hint:
                if not ldim_label in length_info:
                    raise Exception(f'The string {ldim_label} in the list mask_hint is not a LDim label')
            N = len(length_info[ldim_label])

            inst_mask_shape = tuple([max(length_info[ldim_label]) for ldim_label in mask_hint])
            
            def get_shape_at(ldim_idx):
                return (1,) * ldim_idx + (inst_mask_shape[ldim_idx],) + (1,) * (len(inst_mask_shape) - ldim_idx - 1)
            
            def _get_mask_at(inst_idx):
                masks = [(torch.arange(inst_mask_shape[ldim_idx], device=device) < length_info[ldim_label][inst_idx]).reshape(get_shape_at(ldim_idx)).broadcast_to(inst_mask_shape) for ldim_idx, ldim_label in enumerate(mask_hint)]
                return torch.any(torch.stack(masks, dim=0), dim=0)
            
            masks = [_get_mask_at(inst_idx) for inst_idx in range(N)]
            return torch.stack(masks, dim=0) 
    
    def to_device(self, device):
        if self.preset_padder_data is None:
            new_padding = None
        else:
            new_padding = self.preset_padder_data.to_device(device)
        if self.is_grouped:
            new_data_groups = [group.to_device(device) for group in self.data_groups]
            dc = DataCollection(self.shapesig_data, data_groups=new_data_groups, length_info=self.length_info, data_partition=self.data_partition, padding=new_padding)
        else:
            new_data_list = [data.to_device(device) for data in self.data_list]
            dc = DataCollection(self.shapesig_data, data_list=new_data_list, padding=new_padding)
        return dc

    def apply(self, module:AnnotatedModule, mask_hint='copy', input_key_conv=None, output_key_conv=None) -> 'DataCollection':
        """
        """
        if not self.is_grouped:
            raise Exception('Currently can apply AnnotatedModule only when the DataCollection is "grouped"')
        
        output_shapesig_data, mask_hint = module.get_output_shapesig_data_and_mask_hint(self.shapesig_data, input_key_conv=input_key_conv, output_key_conv=output_key_conv)
        
        key_converted_data_groups = [group.keys_converted(input_key_conv) for group in self.data_groups]
        if mask_hint is not None:
            new_data_groups = [TensorData(module(group.value, self.get_mask(i, mask_hint=mask_hint, key_conv=input_key_conv))) for i, group in enumerate(key_converted_data_groups)]
        else:
            new_data_groups = [TensorData(module(group.value)) for group in key_converted_data_groups]
        
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
        
    def __setitem__(self, key:str, sub:'DataCollection'):
        if self.get_group_hash() != sub.get_group_hash():
            raise IncompatibleDataCollectionError(f'It seems two DataCollections are not compatible, so you cannot __setitem__ one into another.')
        if self.is_grouped:
            for group, sub_group in zip(self.data_groups, sub.data_groups):
                group:TensorData
                sub_group:TensorData
                group[key] = sub_group 
            self.length_info.update(sub.length_info)    
        else:
            for data, sub_data in zip(self.data_groups, sub.data_groups):
                data[key] = sub_data
        self.shapesig_data[key] = sub.shapesig_data
    
    @classmethod
    def from_dict(self, d:Dict[str, 'DataCollection']):
        first_dc = list(d.values())[0]
        rest_dc = list(d.values())[1:]
        group_hash = first_dc.get_group_hash()
        for i, dc in enumerate(rest_dc[1:]):
            if dc.get_group_hash() != group_hash:
                raise IncompatibleDataCollectionError(f'It seems {i+2}th DataCollection is incompatible with the first one, so you cannot combine them into a single DataCollection')
            
        new_shapesig_data = ShapeSignatureData.from_data_dict({key: dc.shapesig_data for key, dc in d.items()})
        
        if any([dc.preset_padder_data is not None for dc in d.values()]):
            padding = {}
            for key, dc in d.items():
                if dc.preset_padder_data is not None:
                    padding[key] = dc.preset_padder_data
            padding = PadderData.from_data_dict(padding)
        else:
            padding = None
        
        def _get_merged_length_info():
            merged_length_info = {}
            for dc in d.values():
                for ldim_label, info in dc.length_info.items():
                    if ldim_label in merged_length_info and merged_length_info[ldim_label] != info:
                        raise IncompatibleDataCollectionError(f'length_info collision for ldim_label {ldim_label} - {dc} and something else') #Should we make a new error type? 
                    merged_length_info[ldim_label] = info
            return merged_length_info
        
        if first_dc.is_grouped:
            new_data_groups = [TensorData.from_data_dict({key: dc.data_groups[i] for key, dc in d.items()}) for i in range(len(first_dc.data_groups))]
            
            new_length_info = _get_merged_length_info()
            
            data_partition = first_dc.data_partition
            return DataCollection(new_shapesig_data, data_groups=new_data_groups, length_info=new_length_info, data_partition=data_partition, padding=padding)
        else:
            new_data_list = [TensorData.from_data_dict({key: dc.data_list[i] for key, dc in d.items()}) for i in range(len(first_dc.data_list))]
            
            return DataCollection(new_shapesig_data, data_list=new_data_list, padding=padding)


#testing 
def test_to_device():
    from asym.padding import CDimPadder
    data_list=[{'a':torch.rand(2,3), 'b':torch.tensor(1.0)}, {'aa':torch.rand(4,3), 'b':torch.tensor(2.0)}]
    shape_annot = {'a':'(B,L_len,3)', 'b':'(B, 1)'}
    padding={'a':CDimPadder(torch.tensor([1.,0,0]))}
    dc1 = DataCollection(shape_annot, data_list=data_list, padding=padding)
    dc2 = dc1.to_device(torch.device('cuda:0'))
    print(dc2.data_list[0].value)
    print(dc2.preset_padder_data['a'].value.value)
    
    
if __name__ == '__main__':
    test_to_device()
    