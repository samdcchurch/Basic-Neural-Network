import numpy as np

def sigmoid(Z, derivative = False):
    """
    Returns sigmoid(Z). If derivative == True, then returns derivative of sigmoid(Z).
    Works with numpy matrices.
    """

    S = 1 / (1 + np.exp(-Z))

    if not derivative:
        return S 
    else: 
        return S * (1 - S)
    
def relu(Z, derivative = False):
    """
    Returns ReLU(Z). If derivative == True, then returns derivative of ReLU(Z).
    Works with numpy matrices.
    """

    if not derivative:
        return np.maximum(0, Z)
    else:
        return (Z > 0).astype(Z.dtype)
