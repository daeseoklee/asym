from time import time
from copy import deepcopy

import torch
import torch.nn as nn 

from asym.padding import CDimPadder
from asym.grouper import UniGrouper, LengthThresholdGrouper
from asym.data import TensorData
from asym.data_collection import DataCollection, IncompatibleDataCollectionError
from asym.annotated_module import AnnotatedModule

class Example1StrangeModule(AnnotatedModule):
    def __init__(self, dim1, dim2, dim=8):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim = dim 
        self.linear1 = nn.Linear(dim1, dim)
        self.linear2 = nn.Linear(dim2, dim)
    def forward(self, d):
        x1 = d['objects1']['feature']
        v1 = d['objects1']['position']
        x2 = d['objects2']['feature']
        v2 = d['objects2']['position']
        assert v1.shape[-1] == v2.shape[-1] == 3
        N = x1.shape[0]
        assert N == v1.shape[0] == x2.shape[0] == v2.shape[0]
        assert x1.shape[-1] == self.dim1 and x2.shape[-1] == self.dim2
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

        shape1 = x1.shape[1:-1]
        shape2 = x2.shape[1:-1]
        one1 = tuple([1 for _ in range(len(shape1))])
        one2 = tuple([1 for _ in range(len(shape2))])
        x1_reshaped = x1.reshape((N,)+shape1+one2+(self.dim,)).broadcast_to((N,)+shape1+shape2+(self.dim,))
        x2_reshaped = x2.reshape((N,)+one1+shape2+(self.dim,)).broadcast_to((N,)+shape1+shape2+(self.dim,))
        v1_reshaped = v1.reshape((N,)+shape1+one2+(3,)).broadcast_to((N,)+shape1+shape2+(3,))
        v2_reshaped = v2.reshape((N,)+one1+shape2+(3,)).broadcast_to((N,)+shape1+shape2+(3,))
        dot_product = torch.sum(x1_reshaped * x2_reshaped, dim=-1)
        squared_dist = torch.sum((v1_reshaped - v2_reshaped) ** 2, dim=-1)
        return dot_product * squared_dist
    def requires_mask(self):
        return False
    def get_input_annot(self):
        return {
            'objects1': {
                'feature': '(b, .., m_1)',
                'position': '(b, .., 3)'
            },
            'objects2': {
                'feature': '(b, .2., m_2)',
                'position': '(b, .2., 3)'
            }
        }
    def get_output_annot(self):
        return '(b, .., .2.)'

def example1():

    a = time()
    shape_annot = {
        'protein': {
            'p': '(B, L_res, M_1)',
            'R': '(B, L_res, 3, 3)',
            't': '(B, L_res, 3)'
        },
        'ligand': {
            'm': '(B, L_atm, M_2)',
            't': '(B, L_atm, 3)'
        }
    }
    
    identity_matrix = torch.tensor([[1.,0,0],[0,1,0],[0,0,1]])
    padding = {
        'protein' : {
            'R': CDimPadder(identity_matrix)
        }
    }
    
    data_list = []
    length_pairs = [(10, 3), (20, 6), (12, 20)]
    for num_res, num_atm in length_pairs:
        data = TensorData({
        'protein': {
            'p': torch.rand(num_res, 8),
            'R': torch.rand(num_res, 3, 3),
            't': torch.rand(num_res, 3)
        },
        'ligand': {
            'm': torch.rand(num_atm, 4), 
            't': torch.rand(num_atm, 3)
        }
        })  
        data_list.append(data)
    
    print('\n*Initial (len_protein, len_ligand) pairs:')
    for num_res, num_atm in length_pairs:
        print((num_res, num_atm))

    dc = DataCollection(shape_annot, data_list=data_list)
    dc.group(grouper = LengthThresholdGrouper('res', [15]), padding=padding)
    
    g = dc.data_groups[0]
    print('\n* first grouping:')
    print('partition:', dc.data_partition)
    print('first group shape of "p":', g['protein']['p'].value.shape)
    print('first group shape of "R":', g['protein']['R'].value.shape)
    print('first group shape of "m":', g['ligand']['m'].value.shape)
    #print(g['protein']['R'].value[:, -3:]) #Observe the identity paddings
    

    dc.regroup(grouper = LengthThresholdGrouper('atm', [10]))
    
    g = dc.data_groups[0]
    print('\n* second grouping:')
    print('partition:', dc.data_partition)
    print('first group shape of "p":', g['protein']['p'].value.shape)
    print('first group shape of "R":', g['protein']['R'].value.shape)
    print('first group shape of "m":', g['ligand']['m'].value.shape)
    

    prot_dc = dc['protein']
    g = prot_dc.data_groups[0]
    print('\n* In __getitem__("protein"):')
    print('first group shape of "p":', g['p'].value.shape)
    print('first group shape of "R":', g['R'].value.shape)
    
    dc['duplicate'] = deepcopy(dc)
    print('\n* After __setitem__("duplicate")')
    print(dc.shapesig_data.get_template().value) #Observe added keys

    try:
        another_dc = deepcopy(dc)
        another_dc.regroup(UniGrouper())
        dc['another_duplicate'] = another_dc 
    except IncompatibleDataCollectionError as e:
        print('\nThis __setitem__() is invalidated since the two collections are incompatible')

    key_conv = {
        'protein': ('objects1', {
            'p': ('feature', None),
            't': ('position', None)
        }),
        'ligand': ('objects2', {
            'm': ('feature', None),
            't': ('position', None)
        })
    }
    #The following computation is done in per-group basis 
    result = dc.apply(Example1StrangeModule(8, 4), input_key_conv=key_conv, require_mask=False) 
    result.ungroup() 
    print('\n* Shapes after applying Example1StrangeModule():')
    for data in result.data_list:
        print(data.value.shape) 
    b = time()
    print(f'\ntime: {b-a} seconds')
    
    
if __name__ == '__main__':
    example1()