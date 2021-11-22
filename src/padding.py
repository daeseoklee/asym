from typing import Tuple, List, Dict, Any, Callable, Union
from data import Data, DataTemplate
import torch
from torch import Tensor 
from shape_signature import ShapeSignature
from abc import ABCMeta, abstractmethod
from torch.nn.functional import pad as _pad
from copy import deepcopy

class Padder(metaclass=ABCMeta):
    def __init__(self):
        pass
    @abstractmethod 
    def pad(self, t:Tensor, shapesig: ShapeSignature, target_shape:Tuple):
        pass

class ZeroPadder(Padder):
    def __init__(self):
        super().__init__()
    def pad(self, t:Tensor, shapesig: ShapeSignature, target_shape:Tuple):
        if len(t.shape) != len(target_shape):
            raise Exception('shape length does not match')
        nums = []
        for i in range(len(target_shape) - 1, -1, -1):
            nums += [0, target_shape[i] - t.shape[i]]
        return _pad(t, tuple(nums), mode='constant', value=0) 

class CDimPadder(Padder):
    def __init__(self, value:Tensor):
        super().__init__()
        self.value = value
    def pad(self, t:Tensor, shapesig: ShapeSignature, target_shape:Tuple):
        assert t.device == self.value.device
        cdim_indices = list(shapesig.iter_cdim_idx())
        non_cdim_indices = [i for i in range(len(target_shape)) if not i in cdim_indices]
        for i, idx in enumerate(cdim_indices):
            assert t.shape[idx] == target_shape[idx] == self.value.shape[i]
        cdim_shape = tuple([target_shape[i] if i in cdim_indices else 1 for i in range(len(target_shape))])
        v = self.value.reshape(cdim_shape)
        for i in non_cdim_indices:
            broadcast_shape = t.shape[:i] + (target_shape[i] - t.shape[i],) + t.shape[i + 1:]
            t = torch.cat([t, v.broadcast_to(broadcast_shape)], dim=i)
        return t
        
    

    
class PadderData(Data):
    def __init__(self, padding:Union[Padder, Dict[str, Any]], template:DataTemplate=None):
        if template is not None:
            padding = PadderData.padding_with_default_added(padding, template) 
        super().__init__(padding)
    
    @classmethod
    def padding_with_default_added(cls, padding:Union[Padder, Dict[str, Any]], template:DataTemplate):
        
        if template.is_leaf:
            if padding is None:
                return ZeroPadder()
            else:
                assert Padder in type(padding).mro()
                return padding 
        new_padding = {}
        for key in template.keys():
            if padding is None or not key in padding:
                new_padding[key] = cls.padding_with_default_added(None, template[key])
            else:
                new_padding[key] = cls.padding_with_default_added(padding[key], template[key])
        return new_padding

        
        
        
    @property
    def leaf_type(self):
        return Padder


def merge_tensors(tensors:List[Tensor], shapesig: ShapeSignature, padder:Padder) -> Tensor:
    #This affects tensors
    prestack_shapesig = shapesig.bdim_removed()
    assert len(tensors) >= 1
    target_shape = tensors[0].shape
    for idx, _ in prestack_shapesig.iter_ldim_idx_label():
        max_len = -1
        for t in tensors:
            if t.shape[idx] > max_len:
                max_len = t.shape[idx]
        target_shape = target_shape[:idx] + (max_len,) + target_shape[idx + 1:]
    padded_tensors = [padder.pad(t, prestack_shapesig, target_shape) for t in tensors]
    return torch.stack(padded_tensors, dim=shapesig.bdim_idx)

#testing------------------------------------

def test_cdim_padder():
    R = torch.rand(2,1,3,3)
    R_sig = ShapeSignature.parse('(B, L_1, 3, 3)') #(batch, length, 3, 3)
    identity = torch.tensor([[1.,0,0],[0,1,0],[0,0,1]])
    identity_padder = CDimPadder(identity)
    padded = identity_padder.pad(R, R_sig, (2, 4, 3, 3))
    print(padded)


#------------------------------------------

if __name__ == '__main__':
    test_cdim_padder()