from elastic_energy import NeoHookeanElasticEnergy
from fem_system import FEMSystem, compute_shape_matrices
import torch

class ObjectiveBV:
    def __init__(self, vt_surf, bv):
        self.vt_surf, self.bv= vt_surf, bv
        self.length_scale = torch.linalg.norm(vt_surf.max(dim=0).values - vt_surf.min(dim=0).values)
    
    def obj(self, v):
        return objective_target_BV(v, self.vt_surf, self.bv) / (self.length_scale ** 2)
    
    def grad(self, v):
        return grad_objective_target_BV(v, self.vt_surf, self.bv) / (self.length_scale ** 2)
    
class ObjectiveReg:
    def __init__(self, params_init, params_idx, harm_int, weight_reg=0.0, energy_scale=1.0):
        self.params_init, self.params_idx, self.harm_int = params_init, params_idx, harm_int
        self.weight_reg, self.energy_scale = weight_reg, energy_scale
    
    def obj(self, solid, params_tmp):
        if self.weight_reg == 0: return 0
        return self.weight_reg / self.energy_scale * regularization_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)
    
    def grad(self, solid, params_tmp):
        if self.weight_reg == 0: return torch.zeros_like(self.params_init)
        return self.weight_reg / self.energy_scale * regularization_grad_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)

def objective_target_BV(v, vt, bv):
    '''
    Args:
        v: torch Tensor of shape (#v, 3), containing the current vertices position
        vt: torch Tensor of shape (#bv, 3), containing the target surface 
        bv: boundary vertices index (#bv,)
    
    Returns:
        objective: single scalar measuring the deviation from the target shape
    '''
    return 0

def grad_objective_target_BV(v, vt, bv):
    '''
    Args:
        v: torch Tensor of shape (#v, 3), containing the current vertices position
        vt: torch Tensor of shape (#bv, 3), containing the target surface 
        bv: boundary vertices (#BV,)
    
    Returns:
        gradient : torch Tensor of shape (#v, 3)
    '''
    return torch.zeros_like(v)


def regularization_neo_hookean(params_prev, solid, params, params_idx, harm_int):
    '''
    Input:
    - params_prev : array of shape (3*#params,) containing the previous shape
    - solid       : a FEM system to copy
    - params      : array of shape (3*#params,) containing the new shape parameters
    - params_idx  : parameters index in the vertex list. Has shape (#params,)
    - harm_int    : an harmonic interpolator
    
    Output:
    - energy    : the neo hookean energy
    '''
    
    v_prev = solid.v_rest.clone().detach()
    v_prev[params_idx, :] = params_prev.reshape(-1, 3)
    v_prev = harm_int.interpolate_fill(v_prev)

    f_mass = torch.zeros(size=(3,))

    ee_tmp        = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
    solid_virtual = FEMSystem(v_prev, solid.tet, ee_tmp, rho=solid.rho, 
                                 pin_idx=solid.pin_idx, f_mass=f_mass)
    
    v_new = solid_virtual.v_rest.detach()
    v_new[params_idx, :] = params.reshape(-1, 3)
    v_new = harm_int.interpolate_fill(v_new)

    # Compute energy of the virtual solid
    jac = solid_virtual.compute_jacobians(v_new)
    strain = solid_virtual.compute_strain_tensor(jac)
    energy = solid_virtual.compute_elastic_energy(jac, strain)
    
    return energy

def regularization_grad_neo_hookean(params_prev, solid, params, params_idx, harm_int):
    '''
    Input:
    - params_prev : array of shape (3*#params,) containing the previous shape
    - solid       : a FEM system to copy
    - params      : array of shape (3*#params,) containing the new shape parameters
    - params_idx  : parameters index in the vertex list. Has shape (#params,)
    - harm_int    : an harmonic interpolator
    
    Output:
    - grad_reg    : array of shape (3*#params,), the regularization gradient
    '''

    v_prev = solid.v_rest.clone().detach()
    v_prev[params_idx, :] = params_prev.reshape(-1, 3)
    v_prev = harm_int.interpolate_fill(v_prev)

    ee_tmp        = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
    solid_virtual = FEMSystem(v_prev, solid.tet, ee_tmp, rho=solid.rho, 
                                 pin_idx=solid.pin_idx, f_mass=solid.f_mass)
    
    v_new = solid_virtual.v_rest.detach()
    v_new[params_idx, :] = params.reshape(-1, 3)
    v_new = harm_int.interpolate_fill(v_new)

    # Compute energy of the virtual solid
    jac = solid_virtual.compute_jacobians(v_new)
    strain = solid_virtual.compute_strain_tensor(jac)
    f_el = solid_virtual.compute_elastic_forces(jac, strain)
    
    # Negative of the elastic forces: gradient of the energy
    grad_reg = - f_el[params_idx].reshape(-1,)
    return grad_reg