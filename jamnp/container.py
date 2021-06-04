import itertools
from typing import List, Union

import numpy as np


def fstack(arrays:List[Union[List,np.ndarray]])->np.ndarray:
    """stack list/ndarray of different size into ndarray with zero padding

    :param arrays: [description]
    :type arrays: List[Union[List,np.ndarray]]
    :return: stack array
    :rtype: np.ndarray
    """
    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    # find longest row length
    row_length = max(arrays, key=len).__len__()
    mat = np.array( [resize(row, row_length) for row in arrays] )

    return mat

def fstack_pad(arrays:List[Union[List,np.ndarray]],fillvalule:float=0)->np.ndarray:
    """stack list/ndarray of different size into ndarray with zero padding

    :param arrays: 
    :type arrays: List[Union[List,np.ndarray]]  
    :param fillvalule: padding value, defaults to 0
    :type fillvalule: float, optional
    :return: stack array
    :rtype: np.ndarray
    """
    return np.column_stack((itertools.zip_longest(*arrays, fillvalue=fillvalule)))
