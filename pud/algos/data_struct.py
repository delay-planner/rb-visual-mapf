"""
Helper functions for data structures
"""
from typing import Dict, List, Union
import torch
import numpy as np
def inp_to_device(
        inp:Union[np.ndarray, Dict[str, np.ndarray], Dict[str, torch.Tensor]], 
        device:torch.device,
        ):
    """convert dict inps to torch"""
    if isinstance(inp, dict):
        for key in inp:
            if isinstance(inp[key], np.ndarray):
                inp[key] = torch.from_numpy(inp[key]).to(device)
            elif isinstance(inp[key], torch.Tensor):
                inp[key] = inp[key].to(device)
            else:
                raise Exception("data type mismatch")
        return inp
    
    if isinstance(inp, np.ndarray):
        inp = torch.from_numpy(inp).to(device)
    elif isinstance(inp, torch.Tensor):
        inp = inp.to(device)
    else:
        raise Exception("data type mismatch")
    return inp

def init_embedded_dict(D:dict, embeds:List[tuple]=[]):
    """
    in-place init of embedded dict
    the init function should be either a list or dict
    embeds = [(key, init_function), ...]

    example: 
    DD = {}
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    """
    tmp_D = D
    for next_key, init_f in embeds:
        if not (next_key in tmp_D):
            tmp_D[next_key] = init_f()
        
        assert isinstance(tmp_D[next_key], init_f)
        tmp_D = tmp_D[next_key]
    return 