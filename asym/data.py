from typing import Tuple, List, Dict, Any, Callable, Union, Generator
from abc import ABCMeta, abstractmethod

import torch 
from torch import Tensor
from torch.nn.functional import pad


class Data:
    def __init__(self, value:Union[Any, Dict[str, Any]]):
        self.value = value
        if type(value) == dict:
            self.is_leaf = False
        else:   
            self.is_leaf = True
    
    @classmethod
    def from_data_dict(cls, data_dict:Dict[str, Any]):
        return cls({key: data.value for key, data in data_dict.items()})
            
    def __getitem__(self, key:str) -> 'Data':
        if self.is_leaf:
            raise Exception('The data is a leaf')
        cls = type(self)
        return cls(self.value[key])
    
    def __setitem__(self, key:str, sub:'Data'):
        if self.is_leaf:
            raise Exception('The data is a leaf')
        if type(sub) != type(self):
            raise Exception()
        self.value[key] = sub.value
    
    def __contains__(self, key:str):
        return key in self.value.keys()
    
    def __iter__(self, key:str):
        return self.value.keys()
    
    def keys(self):
        return list(self.value.keys())
    
    @property
    def leaf_type(self):
        pass
    def get(self, keys:List[str]) -> 'Data':
        if keys == []:
            return self
        return self[keys[0]].get(keys[1:])
    
    def get_any_leaf_keys(self, acc=[]) -> List[str]:
        if self.is_leaf:
            return acc 
        key = self.keys()[0]
        return self[key].get_any_leaf_keys(acc=acc+[key])
    
    def get_any_leaf_value(self) -> Any:
        if self.is_leaf:
            return self.value 
        key = self.keys()[0]
        return self[key].get_any_leaf_value()

    def get_any_leaf_item(self, acc=[]) -> Tuple[List[str], Any]:
        if self.is_leaf:
            return acc, self.value 
        key = self.keys()[0]
        return self[key].get_any_leaf_item(acc=acc+[key])
    
    def get_template(self, max_depth=5):
        def _get_template(d, depth=0):
            if depth > max_depth:
                raise Exception(f'The data dictionary is too deep (depth > {max_depth})')
            if type(d) != dict:
                return None
            return {key: _get_template(val, depth=depth+1) for key, val in d.items()}
        return DataTemplate(_get_template(self.value))
    
    def keys_converted(self, key_conv:Dict[str, Tuple[str, Union[None, Any]]], reversed=False) -> 'Data':
        def get_reversed_key_conv(key_conv):
            if key_conv is None:
                return None 
            return {key2: (key1, get_reversed_key_conv(sub)) for key1, (key2, sub) in key_conv.items()}
        if reversed:
            key_conv = get_reversed_key_conv(reversed)
        def _keys_converted(d:Union[Any, Dict[str, Any]], key_conv:Dict[str, Tuple[str, Union[None, Any]]]):
            if key_conv is None or type(d) != dict:
                return d 
            new_dict = {}
            for key, val in d.items():
                if key in key_conv:
                    new_key, sub_key_conv = key_conv[key]
                    new_dict[new_key] = _keys_converted(val, sub_key_conv)
                else:
                    new_dict[key] = val 
            return new_dict
        return type(self)(_keys_converted(self.value, key_conv))
    
    # to avoid further recursive definitions
    
    @classmethod
    def leaf_zip_left(cls, data1:'Data', data2:'Data') -> Generator[Tuple[Any, Any], None, None]:
        if data1.is_leaf:
            if not data2.is_leaf:
                raise Exception(f'In leaf_zip_left(data1, data2), the template of data2 is not strictly bigger than the template of data1')
            yield (data1.value, data2.value)
            return
        for key in data1.keys():
            if data2.is_leaf or not key in data2.keys():
                raise Exception(f'In leaf_zip_left(data1, data2), the template of data2 is not strictly bigger than the template of data1\ndata1:{data1.value},\ndata2:{data2.value}')
            yield from cls.leaf_zip_left(data1[key], data2[key])
    
    @classmethod 
    def leaf_zip_right(cls, data1:'Data', data2:'Data') -> Generator[Tuple[Any, Any], None, None]:
        if data2.is_leaf:
            if not data1.is_leaf:
                raise Exception(f'In leaf_zip_left(data1, data2), the template of data1 is not strictly bigger than the template of data2')
            yield (data1.value, data2.value)
        for key in data2.keys():
            if data1.is_leaf or not key in data1.keys():
                raise Exception(f'In leaf_zip_left(data1, data2), the template of data1 is not strictly bigger than the template of data2')
            yield from cls.leaf_zip_right(data1[key], data2[key])
    
    @classmethod 
    def map(cls, f:Callable[[Any], Any], data:'Data') -> 'Data':
        def _map(d:Union[Any, Dict[str, Any]]):
            if type(d) == dict:
                return {key: _map(val) for key, val in d.items()}
            return f(d)
        return cls(_map(data.value))
    
    @classmethod 
    def map2(cls, f:Callable[[Any, Any], Any], data1:'Data', data2:'Data') -> 'Data':
        #[cls] is the return type 
        def _map2(d1, d2):
            if type(d1) == dict:
                return {key: _map2(d1[key], d2[key]) for key in d1}
            return f(d1, d2)
        return cls(_map2(data1.value, data2.value))
    
class DataTemplate(Data):
    def __init__(self, value:Union[None, Dict[str, Any]]):
        super().__init__(value)
    @property
    def leaf_type(self):
        return type(None)


class TensorData(Data):
    def __init__(self, value:Union['TensorData', Tensor, Dict[str, Any]]):
        super().__init__(value)
    @property
    def leaf_type(self):
        return Tensor
    def to_device(self, device):
        def _to_device(d):
            if type(d) == dict:
                return {key: _to_device(val) for key, val in d.items()}
            return d.to(device=device)
        return TensorData(_to_device(self.value))

