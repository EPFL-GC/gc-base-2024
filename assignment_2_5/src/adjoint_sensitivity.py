import torch
from utils import linear_solve

torch.set_default_dtype(torch.float64)

def compute_adjoint(solid, jac, E, dJdx_U):
    '''
    This assumes that the solid is at equilibrium when called
    
    Args:
        solid: an elastic solid at equilibrium
        jac: the current jacobian of the deformation (#v, 3, 3)
        E: strain induced by the current deformation (#t, 3, 3), can be None for Neo-Hookean
        dJdx_U: torch tensor of shape (#unpinned, 3) containing the gradient of the objective w.r.t. the unpinned vertices
    
    Returns:
        adjoint: torch tensor of shape (3 * #v,), the adjoint state vector
    '''
    return torch.zeros((jac.shape[0] * 3,))