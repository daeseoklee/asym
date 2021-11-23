from typing import Tuple, List, Dict, Any, Union
from torch import Tensor

from asym.data import Data

class PreDim:
    def __init__(self):
        pass
    def __str__(self):
        pass
    
class Dim:
    def __init__(self):
        pass
    def __str__(self):
        pass

class ADim(PreDim): #abbreviated dimension 
    def __init__(self, label):
        super().__init__()
        self.label = label
    def __str__(self):
        return f'.{self.label}.'
    def __eq__(self, other:PreDim):
        return type(other) == ADim 

class PreBDim(PreDim): #batch dimension 
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'b'
    def __eq__(self, other:PreDim):
        return type(other) == PreBDim 
        
class BDim(Dim): #batch dimension 
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'B'
    def __eq__(self, other:Dim):
        return type(other) == BDim 

class PreMDim(PreDim): #model dimension 
    def __init__(self, label):
        super().__init__()
        self.label = label
    def __str__(self):
        return f'm_{self.label}'
    def __eq__(self, other:PreDim):
        return type(other) == PreMDim and self.label == other.label

class MDim(Dim): #model dimension 
    def __init__(self, label):
        super().__init__()
        self.label = label
    def __str__(self):
        return f'M_{self.label}'
    def __eq__(self, other:Dim):
        return type(other) == MDim and self.label == other.label 

class PreLDim(PreDim): #length dimension
    def __init__(self, label):
        super().__init__()
        self.label = label
    def __str__(self):
        return f'l_{self.label}'
    def __eq__(self, other:PreDim):
        return type(other) == PreLDim and self.label == other.label 

class LDim(Dim): #length dimension
    def __init__(self, label):
        super().__init__()
        self.label = label
    def __str__(self):
        return f'L_{self.label}'
    def __eq__(self, other:Dim):
        return type(other) == LDim and self.label == other.label 

class PreCDim(PreDim): #constant dimension 
    def __init__(self, num):
        super().__init__()
        self.num = num
    def __str__(self):
        return str(self.num)
    def __eq__(self, other:PreDim):
        return type(other) == PreCDim and self.num == other.num 
        
class CDim(Dim): #constant dimension 
    def __init__(self, num):
        super().__init__()
        self.num = num
    def __str__(self):
        return str(self.num)
    def __eq__(self, other:Dim):
        return type(other) == CDim and self.num == other.num 

class PreShapeSignature: 
    def __init__(self, l:List[PreDim], come_from='input'):
        self.list : List[PreDim] = l 
        self.come_from = come_from
        self.validate_list()
        self.bdim_idx = self.find_bdim()

    @classmethod
    def parse(cls, s:str, come_from='input'):
        assert s[0] == '('
        assert s[-1] == ')'
        return cls([cls.parse_atom(x.strip()) for x in s[1:-1].split(',')], come_from=come_from)
    
    @classmethod
    def parse_atom(cls, s:str):
        assert len(s) >= 1
        if s.startswith('.'):
            assert s.endswith('.')
            return ADim(s[1:-1])
        if s == 'b':
            return PreBDim()
        if s.isnumeric():
            return PreCDim(int(s))
        if s[0] == 'm':
            assert s[1] == '_'
            return PreMDim(s[2:])
        if s[0] == 'l':
            assert s[1] == '_'
            return PreLDim(s[2:])
        raise Exception()
    
    def __str__(self):
        s = '('
        for i, predim in enumerate(self.list):
            if i > 0:
                s += ', '
            s += str(predim)
        s += ')'
        return s
    
    def __len__(self):
        return len(self.list)
    
    def validate_list(self):
        if sum([type(pdim) == PreBDim for pdim in self.list]) != 1:
            raise Exception('There must be exactly one batch dimension')
        if self.come_from == 'input' and sum([type(pdim) == ADim for pdim in self. list]) > 1:
            raise Exception('There can be at most one abbreviated dimension')

    def find_bdim(self):
        for i, dim in enumerate(self.list):
            if type(dim) == PreBDim:
                return i
        return -1
    
    def find_adim(self):
        for i, predim in enumerate(self.list):
            if type(predim) == ADim:
                return i
        return None
    
class PreShapeSignatureData(Data):
    def __init__(self, value:Union[PreShapeSignature, Dict[str, Any]]):
        super().__init__(value)
        any_sig = self.get_any_leaf_value()
        any_sig:ShapeSignature
        if any_sig.bdim_idx == -1:
            self.batched = False
        else:
            self.batched = True 

    @property
    def leaf_type(self):
        return PreShapeSignature
    
    @classmethod
    def parse(cls, shape_annot:Dict[str, Union[str, Any]], come_from='input'):
        def _parse_preshapesig_data(shape_annot):
            if type(shape_annot) == str:
                return PreShapeSignature.parse(shape_annot, come_from=come_from)
            assert type(shape_annot) == dict
            return {key: _parse_preshapesig_data(val) for key, val in shape_annot.items()}
        return cls(_parse_preshapesig_data(shape_annot))


class ShapeSignature: 
    def __init__(self, l:List[Dim]):
        self.list : List[Dim] = l
        self.validate_list()
        self.bdim_idx = self.find_bdim()
        
    def bdim_removed(self):
        if self.bdim_idx == -1:
            raise Exception('No Bdim')
        l = self.list[:self.bdim_idx] + self.list[self.bdim_idx + 1:]
        return ShapeSignature(l)

    @classmethod
    def parse(cls, s:str):
        assert s[0] == '('
        assert s[-1] == ')'
        return cls([cls.parse_atom(x.strip()) for x in s[1:-1].split(',')])
    
    @classmethod
    def parse_atom(cls, s:str):
        assert len(s) >= 1
        if s == 'B':
            return BDim()
        if s.isnumeric():
            return CDim(int(s))
        if s[0] == 'M':
            assert s[1] == '_'
            return MDim(s[2:])
        if s[0] == 'L':
            assert s[1] == '_'
            return LDim(s[2:])
        raise Exception()
    
     
    def __str__(self):
        s = '('
        for i, dim in enumerate(self.list):
            if i > 0:
                s += ', '
            s += str(dim)
        s += ')'
        return s
    
    def __len__(self):
        return len(self.list)
    
    def validate_list(self):
        if sum([type(dim) == BDim for dim in self.list]) > 1:
            raise Exception('There must be at most one batch dimension')
        
    def find_bdim(self):
        for i, dim in enumerate(self.list):
            if type(dim) == BDim:
                return i
        return -1
    
    def iter_cdim_idx(self):
        for i, dim in enumerate(self.list):
            if type(dim) == CDim:
                yield i
                
    def iter_ldim_idx_label(self):
        for i, dim in enumerate(self.list):
            if type(dim) == LDim:
                dim:LDim
                yield i, dim.label
    
class ShapeSignatureData(Data):
    def __init__(self, value:Union[ShapeSignature, Dict[str, Any]]):
        super().__init__(value)
        self.ldim_map = self.get_ldim_map()
        any_sig = self.get_any_leaf_value()
        any_sig:ShapeSignature
        if any_sig.bdim_idx == -1:
            self.batched = False
        else:
            self.batched = True 
        if self.batched:
            self.bdim_loc = self.get_bdim_loc()
            self.prestack_ldim_map = self.get_ldim_map(prestack=True)
        else:
            self.bdim_loc = None
            self.prestack_ldim_map = None
    @property
    def leaf_type(self):
        return ShapeSignature
    
    @classmethod
    def parse(cls, shape_annot:Dict[str, Union[str, Any]]):
        def _parse_shapesig_data(shape_annot):
            if type(shape_annot) == str:
                return ShapeSignature.parse(shape_annot)
            assert type(shape_annot) == dict
            return {key: _parse_shapesig_data(val) for key, val in shape_annot.items()}
        return cls(_parse_shapesig_data(shape_annot))

    def get_bdim_loc(self):
        keys, value = self.get_any_leaf_item()
        bdim_idx = value.bdim_idx
        assert bdim_idx != -1
        return keys, bdim_idx
    
    def get_ldim_map(self, prestack=False):
        ldim_map = {} 
        def fill_ldim_map_(ldim_map, d:Union[ShapeSignature, Dict[str, Any]], keys=[], prestack=prestack):
            if type(d) == dict:
                for key in d:
                    fill_ldim_map_(ldim_map, d[key], keys + [key], prestack=prestack)
            else:
                assert type(d) == ShapeSignature
                for idx, label in d.iter_ldim_idx_label():
                    if label in ldim_map:
                        continue 
                    if prestack:
                        if d.bdim_idx == -1:
                            raise Exception('No batch index while it is expected to be there')
                        if d.bdim_idx < idx:
                            idx = idx - 1
                    ldim_map[label] = (keys, idx)
        fill_ldim_map_(ldim_map, self.value, prestack=prestack)
        return ldim_map 
    
    


        
            
    