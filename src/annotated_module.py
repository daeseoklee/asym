from typing import Tuple, List, Dict, Any, Union
from shape_signature import *
from data import Data
import torch
import torch.nn as nn 


class ShapeConflictError(Exception): 
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
    
class PreDimConversionError(Exception): 
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
    
class AnnotationError(Exception): 
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class AnnotatedModule(nn.Module):
    def __init__(self):
        super().__init__()
    @classmethod
    def get_input_annot(cls) -> Union[str, Dict[str, Any]]:
        pass
    @classmethod
    def get_output_annot(cls) -> Union[str, Dict[str, Any]]:
        pass
    @classmethod 
    def get_output_shapesig_data(cls, input_shapesig_data:ShapeSignatureData, input_key_conv=None, output_key_conv=None) -> ShapeSignatureData:
        """
        Description:
            Given input_presig_data(parsed from self.get_input_annot()), output_presig_data(parsed from self.get_output_annot()) and input_shapesig_data, get output_shapesig_data by:
            1. Match every leaf PreShapeSignature of input_presig_data with corresponding leaf ShapeSignature of input_shapesig_data to get "conversion maps" (conv_ldim, conv_mdim, conv_adim). It is possible that we find "conflict" while updating the conversion maps. 
            2. Use the conversion maps to convert leaf PreShapeSignature objects of output_presig_data to ShapeSignature objects, which comprise the final ShapeSignatureData object i.e. output_shapesig_data.  
        Match example:
            In the following, the evolution of conversion maps with respect to a given sequence of pairs(come from corresponding leaves of input_presig_data and input_shapesig_data) (input_presig, input_shapesig) will be illustrated. 
        
            -convert_ldim = {}
            -convert_mdim = {}
            -convert_adim = {}
            1. (b, l_1, l_2, .., m_1), (B, L_width, L_height, 3, M_hidden1)
            -convert_ldim = {'1': 'width', '2': 'height'}
            -convert_mdim = {'1': 'hidden1'}
            -convert_adim = {'': [CDim(3)]}
            2. (b, l_1, l_3, .?., m_2), (B, L_width, L_aug1, M_aug1dim)
            -convert_ldim = {'1': 'width', '2': 'height', '3': 'aug1'}
            -convert_mdim = {'1': 'hidden1', '2': 'aug1dim'}
            -convert_adim = {'': [CDim(3)], '?': []}    
            3. (b, l_2, ..), (B, L_height, M_aug2dim)
            "Conflict" because convert_adim[''] == [CDim(3)] but corresponding subsequence of input_shapesig is [MDim('aug2dim')]. 

        Args:
            input_shapesig_data (ShapeSignatureData): [description]

        Returns:
            ShapeSignatureData: [description]
        """
        input_annot = cls.get_input_annot()
        if input_annot is None:
            raise AnnotationError(f'It seems get_input_annot() is not implemented for {cls}')
        input_presig_data = PreShapeSignatureData.parse(input_annot, come_from='input')

        if input_key_conv is not None:
            input_shapesig_data = input_shapesig_data.keys_converted(input_key_conv)

        output_annot = cls.get_output_annot()
        if output_annot is None:
            raise AnnotationError(f'It seems get_output_annot() is not implemented for {cls}')
        output_presig_data = PreShapeSignatureData.parse(output_annot, come_from='output')

        conv_ldim:Dict[str, str] = {} #PreLDim label -> LDim label 
        conv_mdim:Dict[str, str] = {} #PreMDim label -> MDIm label
        conv_adim:Dict[str, List[Dim]] = {} #ADim label -> Sequence of Dim 
        def update_conv_maps_(presig:PreShapeSignature, shapesig:ShapeSignature):
            nonlocal conv_ldim
            nonlocal conv_mdim 
            nonlocal conv_adim 
            adim_found = False
            adim_len = None 
            for i, predim in enumerate(presig.list):
                if adim_found:
                    j = i + adim_len - 1 #j is the index for shapesig  
                else:
                    j = i
                if j >= len(shapesig):
                    raise ShapeConflictError(f'{shapesig} is too short compared to {presig}')
                dim = shapesig.list[j] 
                if type(predim) == PreBDim:
                    if type(dim) != BDim:
                        raise ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}]')
                elif type(predim) == PreLDim:
                    if type(dim) != LDim:
                        raise ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}]')
                    if predim.label in conv_ldim:
                        if dim.label != conv_ldim[predim.label]:
                            raise ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}] - {predim} was previously resolved to {LDim(conv_ldim[predim.label])}')
                    else:
                        conv_ldim[predim.label] = dim.label 
                elif type(predim) == PreMDim:
                    if type(dim) != MDim:
                        raise ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}]')
                    if predim.label in conv_mdim:
                        if dim.label != conv_mdim[predim.label]:
                            ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}] - {predim} was previously resolved to {MDim(conv_ldim[predim.label])}')
                    else:
                        conv_mdim[predim.label] = dim.label 
                elif type(predim) == PreCDim:
                    if type(dim) != CDim or predim.num != dim.num:
                        raise ShapeConflictError(f'{presig}[{i}] and {shapesig}[{j}]')
                elif type(predim) == ADim:
                    if adim_found:
                        raise AnnotationError(f'There can be at most one abbreviated dimension in {presig}')
                    if predim.label in conv_adim:
                        expected_sequence = conv_adim[predim.label]
                        if j + len(expected_sequence) - 1 >= len(shapesig):
                            raise ShapeConflictError(f'{shapesig} is too short to match the abbreviated part of {presig}, given that the abbreviation has been determined to be {expected_sequence}')
                        if shapesig.list[j:j+len(expected_sequence)] != expected_sequence:
                            expected_sequence_in_shapesig = ShapeSignature(expected_sequence)
                            raise ShapeConflictError(f'The subsequence of {shapesig} corresponding to the abbreviated part of {presig} does not match with the previously determined sequence {expected_sequence_in_shapesig}')
                        adim_len = len(expected_sequence)
                    else:
                        if len(shapesig) - len(presig) + 1 < 0:
                            raise ShapeConflictError(f'{shapesig} is too short compared to {presig}')
                        adim_len = len(shapesig) - len(presig) + 1
                        conv_adim[predim.label] = shapesig.list[j:j+adim_len] 
                    adim_found = True
                else:
                    raise AnnotationError(f'The {predim} in {presig} is strange.')
                     
        for presig, shapesig in Data.leaf_zip_left(input_presig_data, input_shapesig_data):
            update_conv_maps_(presig, shapesig)
        def conv_presig(presig:PreShapeSignature) -> ShapeSignature:
            l = [] 
            for predim in presig.list:
                if type(predim) == PreBDim:
                    l.append(BDim())
                elif type(predim) == PreLDim:
                    if not predim.label in conv_ldim:
                        raise PreDimConversionError('Currently, we do not support introducing new LDims in AnnotatedModule applications. It may change in later versions.')
                    label = conv_ldim[predim.label]
                    l.append(LDim(label))
                elif type(predim) == PreMDim:
                    if not predim.label in conv_mdim:
                        label = predim.label
                    else:
                        label = conv_mdim[predim.label]
                    l.append(MDim(label))
                elif type(predim) == PreCDim:
                    l.append(CDim(predim.num))
                elif type(predim) == ADim:
                    if not predim.label in conv_adim:
                        raise PreDimConversionError(f'The abbreviated dimension in {predim} does not match any abbrviated dimension of the input signature')
                    l = l + conv_adim[predim.label]
                else:
                    raise AnnotationError(f'The {predim} in {presig} is strange.')
            return ShapeSignature(l)
        
        output_shapesig_data = ShapeSignatureData.map(conv_presig, output_presig_data)
        if output_key_conv is not None:
            output_shapesig_data = output_shapesig_data.keys_converted(output_key_conv)
            
        return output_shapesig_data

#test-----------
class TestModule1(AnnotatedModule):
    def __init__(self):
        super().__init__()
    
    def get_input_annot():
        return {
            '2d-feature': '(b, l_1, l_2, .., m_1)', 
            'augs': {
                'aug1': '(b, l_1, l_3, .?., m_2)'
            }
        }
    def get_output_annot():
        return '(b, l_1, .., l_3, .?., m_3)'

class TestModule2(AnnotatedModule):
    def __init__(self):
        super().__init__()
    def get_input_annot():
        return {
            '2d-feature': '(b, l_1, l_2, .., m_1)', 
            'augs': {
                'aug1': '(b, l_1, l_3, .?., m_2)',
                'aug2': '(b, l_2, ..)'
            }
        }
    def get_output_annot():
        return '(b)'
    
def test_annotated_module():
    input_shapesig_data = ShapeSignatureData.parse({
        'image': '(B, L_width, L_height, 3, M_hidden1)',
        'augs': {
            'widthwise-aug': '(B, L_width, L_aug1, M_aug1dim)',
            'heightwise-aug': '(B, L_height, M_aug2dim)'
        }
    })
    input_key_conv = {
        'image': ('2d-feature', None), 
        'augs': ('augs', {
            'widthwise-aug': ('aug1', None),
            'heightwise-aug': ('aug2', None)
        })
    }
    print(TestModule1.get_output_shapesig_data(input_shapesig_data, input_key_conv=input_key_conv).value)
    print()
    try:
        TestModule2.get_output_shapesig_data(input_shapesig_data, input_key_conv=input_key_conv)
    except ShapeConflictError as e:
        print(f'This error was expected: {e}')
    else:
        raise Exception('Unknown error')

if __name__ == '__main__':
    test_annotated_module()