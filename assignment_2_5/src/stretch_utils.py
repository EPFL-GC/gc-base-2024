import torch
from objectives import *
from utils import *

def stretch_from_point(v, stretches, v_idx):
    '''
    Args:
        v: vertices to be deformed (#v, 3)
        stretches: torch tensor of shape (3,) giving the stretching coefficients along x, y, z
        v_idx: vertex index around which the mesh is stretched
    
    Returns:
        v_stretched: deformed vertices (#v, 3)
    '''
    v_stretched = (v - v[v_idx]) @ torch.diag(stretches) + v[v_idx]
    return v_stretched

def report_stretches(solid, v_rest_init, bv, v_target, stretches, lowest_pin_idx):
    '''
    Args:
        solid: an elastic solid to deform
        v_rest_init: reference vertex position to compress/stretch (#v, 3)
        bv: boundary vertices index (#bv,)
        v_target: target boundary vertices position
        stretches: torch tensor of shape (n_stretches,) containing a stretch factors to try out
        lowest_pin_idx: index of the pinned index that has the lowest z coordinate
    
    Returns:
        list_v_rest: list of n_stretches rest vertex positions that have been tried out
        list_v_eq: list of the corresponding n_stretches equilibrium vertex positions
        target_closeness: torch tensor of shape (n_stretches,) containing the objective values for each stretch factor
    '''
    list_v_rest = []
    list_v_eq   = []
    target_closeness = torch.zeros(size=(stretches.shape[0],))
    v_rest_tmp  = v_rest_init.clone()

    for i, stretch in enumerate(stretches):
        # Compute and update the rest shape of the solid
        stretch_diag = torch.tensor([1.0, 1.0, stretch])
        v_rest_tmp[solid.free_idx] = stretch_from_point(v_rest_init, stretch_diag, lowest_pin_idx)[solid.free_idx]
        solid.update_rest_shape(v_rest_tmp)

        # Compute new equilibrium (use the previous equilibrium state as initial guess if available)
        # You may use equilibrium_convergence_report_NCG to find the equilibrium (20 steps and thresh=1 should do)
        if i == 0:
            v_init_guess = solid.v_rest.clone()
        report = equilibrium_convergence_report_NCG(solid, v_init_guess, 20, thresh=1)
        
        # Update guess for next stretch factor
        v_init_guess = report["final_def"].clone()

        # Fill in the 
        list_v_rest.append(solid.v_rest.clone())
        list_v_eq.append(report["final_def"].clone())
        target_closeness[i] = objective_target_BV(report["final_def"], v_target, bv)
    
    return list_v_rest, list_v_eq, target_closeness