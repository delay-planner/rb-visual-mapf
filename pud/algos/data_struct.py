"""
Helper functions for data structures
"""

def init_embedded_dict(D:dict, embeds:list=[]):
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