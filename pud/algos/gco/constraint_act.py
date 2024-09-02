import numpy as np

class PiecewiseCosineScheduler:
    def __init__(self, target_margin:float, limit:float):
        self.target_margin = target_margin
        self.limit = limit
        self.A = limit - target_margin

    def __call__(self, inp:float, symmetric=True):
        ret = 0.0
        if (symmetric) and inp > self.limit + (self.limit-self.target_margin):
            ret = 0.0
        elif (not symmetric) and inp > self.limit:
            ret = 1.0
        elif inp < self.target_margin:
            ret = 0.0
        else:
            ret = 0.5 * np.cos(np.pi/self.A*(inp - self.A - self.target_margin)) + 0.5            
        return ret

class PiecewiseLinearScheduler:
    def __init__(self, target_margin:float, limit:float):
        self.target_margin = target_margin
        self.limit = limit
        self.A = limit - target_margin

    def __call__(self, inp:float, symmetric=False):
        # todo: symmetric is not added (not needed now), just a compat flag
        ret = 0.0
        if inp > self.limit:
            ret = 1.0
        elif inp < self.target_margin:
            ret = 0.0
        else:
            ret = (inp - self.target_margin) / self.A             
        return ret
    

class JumpScheduler:
    def __init__(self, target_margin:float, limit:float):
        self.target_margin = target_margin
        self.limit = limit
        self.A = limit - target_margin

    def __call__(self, inp:float, symmetric=False):
        ret = 0.0
        if inp > self.limit:
            ret = 1.0
        else:
            ret = 0.0 
        return ret

class ConstantZeroScheduler:
    """practically disable constraint activation function"""
    def __init__(self, target_margin:float, limit:float):
        self.target_margin = target_margin
        self.limit = limit
        self.A = limit - target_margin

    def __call__(self, inp:float, symmetric=False):
        return 0.0
