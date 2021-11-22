from time import time

import torch
import torch.nn as nn 

from padding import CDimPadder
from grouper import LengthThresholdGrouper
from data import TensorData
from data_collection import DataCollection 
from annotated_module import AnnotatedModule

class Example1StrangeModule(AnnotatedModule):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass 
    def get_input_annot():
        pass
    def get_output_annot():
        pass

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
    
    data_list = [None, None, None]
    for i, (num_res, num_atm) in enumerate([(10, 3), (20, 6), (12, 20)]):
        data_list[i] = TensorData({
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

    dc = DataCollection(shape_annot, data_list=data_list)
    dc.group(grouper = LengthThresholdGrouper('res', [15]), padding=padding)
    
    g = dc.data_groups[0]
    print('\n* first grouping:')
    print('partition:', dc.data_partition)
    print('first group shape of "p":', g['protein']['p'].value.shape)
    print('first group shape of "R":', g['protein']['R'].value.shape)
    print('first group shape of "m":', g['ligand']['m'].value.shape)
    #print(g['protein']['R'].value[:, -3:]) #Observe the identity paddings
    

    dc.ungroup()
    dc.group(grouper = LengthThresholdGrouper('atm', [10]))
    
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
    
    
    b = time()
    print(f'\ntime: {b-a} seconds')
    
    
if __name__ == '__main__':
    example1()