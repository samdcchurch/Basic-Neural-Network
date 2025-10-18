import numpy as np

def BCE(AL, Y, m, derivative=False):
    """
    Binary Cross-Entropy Loss.
    
    AL: activations of output layer (shape: [n_output, m])
    Y: true labels (same shape)
    m: number of examples
    derivative: if True, return dAL; else return scalar loss
    """
    if not derivative:
        return -(1/m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    else:
        return -(Y / (AL + 1e-8)) + ((1 - Y) / (1 - AL + 1e-8))
