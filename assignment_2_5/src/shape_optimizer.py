from adjoint_sensitivity import compute_adjoint
from elastic_energy import LinearElasticEnergy, NeoHookeanElasticEnergy
from fem_system import FEMSystem
from harmonic_interpolator import HarmonicInterpolator
from objectives import ObjectiveBV
from objectives import ObjectiveReg
from utils import compute_inverse_approximate_hessian_matrix, get_boundary_and_interior, to_numpy
from vis_utils import get_plot, plot_torch_solid
from IPython import display
import time
import numpy as np
import torch
import os
torch.set_default_dtype(torch.float64)


class ShapeOptimizer():
    def __init__(self, solid, v_eq, vt_surf, bv, be, beTarget, weight_reg=0.0, force_thresh=1.0):
        
        # Elastic Solid with the initial rest vertices stored 
        self.solid = solid

        # Mesh info of solid
        self.bv = bv
        self.iv = np.array([idx for idx in range(self.solid.v_rest.shape[0]) if not idx in bv]).astype(np.int64)
        self.be = be
        self.beTarget = beTarget

        # Initialize Laplacian/Harmonic Interpolator
        v_init_rest   = self.solid.v_rest.clone().detach()
        self.harm_int = HarmonicInterpolator(v_init_rest, self.solid.tet, self.iv)

        # Initialize interior vertices with harmonic interpolation
        self.v_init_rest = self.harm_int.interpolate(v_init_rest[bv])
        self.solid.update_rest_shape(self.v_init_rest)

        # Define optimization params and their indices
        self.params_idx = torch.tensor(np.intersect1d(to_numpy(self.solid.free_idx), bv))
        params_init = v_init_rest[self.params_idx].reshape(-1,)
        self.params, self.params_prev = params_init.clone(), params_init.clone() # At time step t, t-1

        # Target surface and Objectives
        self.energy_scale = torch.abs(solid.ee.young * torch.sum(solid.W0))
        self.vt_surf = vt_surf
        self.weight_reg = weight_reg
        self.tgt_fit = ObjectiveBV(vt_surf, bv)
        self.neo_reg = ObjectiveReg(params_init.clone().detach(), self.params_idx, self.harm_int, weight_reg=weight_reg, energy_scale=self.energy_scale)

        # Compute equilibrium deformation
        self.force_thresh = force_thresh
        self.v_eq = v_eq.clone()
        self.v_eq = self.solid.find_equilibrium(self.v_eq, thresh=self.force_thresh)
        obj_init = self.tgt_fit.obj(self.v_eq.detach())
        print("Initial objective: {:.2e}".format(obj_init.item()))

        # Initialize grad
        self.grad = torch.zeros(size=(3 * self.params_idx.shape[0],))

        # BFGS book-keeping
        self.invB      = torch.eye(3 * self.params_idx.shape[0])
        self.grad_prev = torch.zeros(size=(3 * self.params_idx.shape[0],))

        if "JPY_PARENT_PID" in os.environ:
            self.reset_plot()
        
    def reset_plot(self):
        # Plot the resulting mesh
        rot = torch.tensor(
            [[1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
                [0.0, -1.0, 0.0]]
        )
        aabb = torch.max(self.v_eq, dim=0).values - torch.min(self.v_eq, dim=0).values
        length_scale = torch.mean(aabb)
        self.plot = get_plot(self.solid, self.v_eq, self.be, rot, length_scale, target_mesh=self.vt_surf, be_target=self.beTarget)


    def compute_obj(self):
        '''
        Compute Objective at the current params
        
        Returns:
            obj: Accumulated objective value
        '''

        self.target_fit = self.tgt_fit.obj(self.v_eq.clone().detach())
        self.reg = self.neo_reg.obj(self.solid, self.params)
        obj = self.target_fit + self.reg
        return obj
        
    def compute_gradient(self):
        '''
        Computes the full gradient including the forward simulation and regularization.

        Updated attributes:
            grad: torch.tensor of shape (#params,)
        '''

        # dJ/dx from Target Fitting
        dJ_dx = self.tgt_fit.grad(self.v_eq.clone().detach())[self.solid.free_idx]

        self.grad = gradient_helper_autograd(self.solid, self.v_eq, dJ_dx, self.params, self.params_idx, self.harm_int)
        
        # Add regularization gradient
        self.grad = self.grad + self.neo_reg.grad(self.solid, self.params)
        return self.grad
    
    def update_BFGS(self):
        '''
        Update BFGS hessian inverse approximation
        
        Updated attributes:
            invB: torch.tensor of shape (#params, #params)
        '''
        sk = self.params - self.params_prev
        yk = self.grad - self.grad_prev
        self.invB = compute_inverse_approximate_hessian_matrix(sk.reshape(-1, 1), yk.reshape(-1, 1), self.invB)
    
    def reset_BFGS(self):
        '''
        Reset BFGS hessian inverse approximation to Identity
        
        Updated attributes:
            invB: torch.tensor of shape (#params, #params)
        '''
        self.invB = torch.eye(3 * self.params_idx.shape[0])

    def set_params(self, params_in, verbose=False):
        '''
        Set optimization params to the input params_in
        
        Args:
            params_in: Input params to set the solid to, torch.tensor of shape (#params,)
            verbose: Boolean specifying the verbosity of the equilibrium solve
        
        Updated attributes:
            solid: Elastic solid, specifically, the rest shape and consequently the equilibrium deformation
        '''
        return
        

    def line_search_step(self, step_size_init, max_l_iter):
        '''
        Perform line search to take a step in the BFGS descent direction at the current optimization state
        
        Args:
            step_size_init: Initial value of step size
            max_l_iter: Maximum iterations of line search
        
        Updated attributes:
            solid: Elastic solid, specifically, the rest shape and consequently the equilibrium deformation
            params, params_prev: torch Tensor of shape (#params,)
            grad_prev: torch Tensor of shape (#params,)

        Returns:
            obj_new: New objective value after taking the step
            l_iter: Number of line search iterations taken
            sucess (bool): Whether the line search was successful
        '''
        return 0, 0, False

    def optimize(self, step_size_init=1.0e-3, max_l_iter=10, n_optim_steps=10):
        '''
        Run BFGS to optimize over the objective J.
        
        Args:
            step_size_init: Initial value of step size
            max_l_iter: Maximum iterations of line search
        
        Updated attributes:
            objectives: Tensor tracking objectives across optimization steps
            grad_mags: Tensor tracking norms of the gradient across optimization steps
        '''
        self.objectives  = ...
        ...
        
        startTime = time.time()

        for i in range(n_optim_steps):
            # TODO: Update the gradients

            # TODO: Update quatities of BFGS
           
            # Line Search
            obj_new, l_iter, success = self.line_search_step(step_size_init, max_l_iter)
            
            display.clear_output(wait=True)
            # Remaining time
            curr_time = (time.time() - startTime)
            rem_time  = (n_optim_steps - i - 1) / (i + 1) * curr_time
            print("Objective after {} optimization step(s): {:.4e}".format(i+1, self.objectives[i+1]))
            print("    Line search Iters: " + str(l_iter))
            print("Elapsed time: {:.1f}s. \nEstimated remaining time: {:.1f}s\n".format(curr_time, rem_time))
            
            # Plot the resulting mesh
            rot = torch.tensor(
                [[1.0,  0.0, 0.0],
                 [0.0,  0.0, 1.0],
                 [0.0, -1.0, 0.0]]
            )
            aabb = torch.max(self.v_eq, dim=0).values - torch.min(self.v_eq, dim=0).values
            length_scale = torch.mean(aabb)
            plot_torch_solid(self.solid, self.v_eq, self.be, rot, length_scale, iteration = i, target_mesh=self.vt_surf, be_target=self.beTarget, plot = self.plot)

def gradient_helper_autograd(solid, v_eq, dJ_dx, params_tmp, params_idx, harm_int):
    '''
    Computes a pytorch computational flow to compute the gradient of the forward simulation through automatic differentiation
    
    Args:
        solid: an elastic solid to use for the forces computation
        v_eq: vertex positions at equilibrium, torch Tensor of shape (#v, 3)
        dJ_dx: gradient of the objective w.r.t. the deformed vertices (3*#v,)
        params_tmp: torch Tensor of shape (3*#params,), current optimization parameters
        params_idx: parameters index in the vertex list. Has shape (#params,)
        harm_int: harmonic interpolator
        
    Returns:
        dJ_dX: torch Tensor of shape (3*#params,), gradient of the objective w.r.t. the optimization parameters
    '''
    # Adjoint state y
    jac_eq = solid.compute_jacobians(v_eq)
    E_eq = solid.compute_strain_tensor(jac_eq)
    adjoint = compute_adjoint(solid, jac_eq, E_eq, dJ_dx)    

    # Define the variable to collect the gradient of f_tot. y
    params_collect = params_tmp.clone()
    params_collect.requires_grad = True
    
    # Model f_tot.y as a differentiable pytorch function of params to collect gradient from auto_grad
    dot_prod = adjoint_dot_forces(params_collect, solid, v_eq, adjoint, params_idx, harm_int)
    dot_prod.backward()
    dJ_dX = params_collect.grad.clone()
    params_collect.grad = None # Reset grad

    return dJ_dX
    
def adjoint_dot_forces(params, solid, v_eq, adjoint, params_idx, harm_int):
    '''
    Args:
        params: array of shape (3*#params,)
        solid: an elastic solid to copy with the deformation at equilibrium
        v_eq: vertex positions at equilibrium, torch Tensor of shape (#v, 3)
        adjoint: array of shape (3*#v,)
        params_idx: parameters index in the vertex list. Has shape (#params,)
        harm_int: harmonic interpolator
    
    Returns:
        ot_prod: dot product between forces at equilibrium and adjoint state vector (detached on purpose)
    '''
    
    # From params, compute the full rest state using the harmonic interpolation
    v_vol    = solid.v_rest.detach()
    v_vol[params_idx, :] = params.reshape(-1, 3)
    v_update = harm_int.interpolate_fill(v_vol)

    #Initialize a solid with this rest state and the same deformed state of the solid
    
    if "LinearElasticEnergy" in str(type(solid.ee)):
        ee_tmp   = LinearElasticEnergy(solid.ee.young, solid.ee.poisson)
    elif "NeoHookeanElasticEnergy" in str(type(solid.ee)):
        ee_tmp   = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
        
    solid_tmp = FEMSystem(v_update, solid.tet, ee_tmp, rho=solid.rho, 
                          pin_idx=solid.pin_idx, f_mass=solid.f_mass)

    jac_eq = solid_tmp.compute_jacobians(v_eq)
    E_eq = solid_tmp.compute_strain_tensor(jac_eq)
    _, f_ext_eq = solid_tmp.compute_volumetric_and_external_forces()
    f_el_eq = solid_tmp.compute_elastic_forces(jac_eq, E_eq)
    
    return adjoint.detach() @ (f_el_eq + f_ext_eq).reshape(-1,)

            
    
    

