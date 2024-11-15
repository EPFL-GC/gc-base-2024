from fem_system import compute_barycenters
import igl
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

torch.set_default_dtype(torch.float64)

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

def get_boundary_and_interior(vlen, t):
    bv = np.unique(igl.boundary_facets(t))
    vIdx  = np.arange(vlen)
    iv  = vIdx[np.invert(np.in1d(vIdx, bv))]
    return bv, iv

def linear_solve(L_method, b):
    '''
    Solves Ax = b where A is positive definite.
    A has shape (n, n), x and b have shape (n,).

    Args:
        L_method: a method that takes x and returns the product Ax
        b: right hand side of the linear system

    Returns:
        x_star: torch tensor of shape (n,) solving the linear system
    '''
    dim = b.shape[0]
    def LHSnp(x):
        return to_numpy(L_method(torch.tensor(x)))
    LHS_op = LinearOperator((dim, dim), matvec=LHSnp)
    x_star_np, _  = cg(LHS_op, to_numpy(b))
    x_star = torch.tensor(x_star_np)
    return x_star

def compute_inverse_approximate_hessian_matrix(sk, yk, invB_prev):
    '''
    Args:
        sk: previous step x_{k+1} - x_k, shape (n, 1)
        yk: grad(f)_{k+1} - grad(f)_{k}, shape (n, 1)
        invB_prev: previous Hessian estimate Bk, shape (n, n)
    
    Returns:
        invB_new: previous Hessian estimate Bk, shape (n, n)
    '''
    # Tests sk and yk shape and fixes it if needed : should be (n,1)
    n = invB_prev.shape[0]
    if sk.shape != (n,1):
        sk = sk.reshape(n,1)
    if yk.shape != (n,1):
        yk = yk.reshape(n,1)

    invB_new  = invB_prev.clone()
    invB_new += (sk.T @ yk + yk.T @ invB_prev @ yk) / ((sk.T @ yk) ** 2) * (sk @ sk.T)
    prod      = (invB_prev @ yk) @ sk.T
    invB_new -= (prod + prod.T) / (sk.T @ yk)
    return invB_new


def equilibrium_convergence_report_NCG(solid, v_init, n_steps, thresh=1.0e-3):
    '''
    Finds the equilibrium by minimizing the total energy using Newton CG.

    Args:
        solid: an elastic solid to optimize
        v_init: the initial guess for the equilibrium position
        n_step: number of optimization steps
        thresh: threshold to stop the optimization process on the gradient's magnitude

    Returns:
        report: a dictionary containing various quantities of interest
    '''

    energies_el  = torch.zeros(size=(n_steps+1,))
    energies_ext = torch.zeros(size=(n_steps+1,))
    residuals    = torch.zeros(size=(n_steps+1,))
    times        = torch.zeros(size=(n_steps+1,))
    
    v_tmp = v_init.clone()
    jac_tmp = solid.compute_jacobians(v_tmp)
    strain_tmp = solid.compute_strain_tensor(jac_tmp)
    def_barycenters_tmp = compute_barycenters(v_tmp, solid.tet)
    f_vol_tmp, f_ext_tmp = solid.compute_volumetric_and_external_forces()
    f_el_tmp = solid.compute_elastic_forces(jac_tmp, strain_tmp)
    
    energies_el[0]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
    energies_ext[0] = solid.compute_external_energy(v_tmp, def_barycenters_tmp, f_vol_tmp)
    residuals[0]    = torch.linalg.norm((f_el_tmp + f_ext_tmp)[solid.free_idx, :])
    idx_stop        = n_steps

    t_start = time.time()
    for i in range(n_steps):
        # Take a Newton step
        v_tmp, jac_tmp, strain_tmp = solid.equilibrium_step(v_tmp, jac_tmp, strain_tmp)
        def_barycenters_tmp = compute_barycenters(v_tmp, solid.tet)
        f_vol_tmp, f_ext_tmp = solid.compute_volumetric_and_external_forces()
        f_el_tmp = solid.compute_elastic_forces(jac_tmp, strain_tmp)

        # Measure the force residuals
        energies_el[i+1]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
        energies_ext[i+1] = solid.compute_external_energy(v_tmp, def_barycenters_tmp, f_vol_tmp)
        residuals[i+1]    = torch.linalg.norm((f_el_tmp + f_ext_tmp)[solid.free_idx, :])
        
        if residuals[i+1] < thresh:
            residuals[i+1:]    = residuals[i+1]
            energies_el[i+1:]  = energies_el[i+1]
            energies_ext[i+1:] = energies_ext[i+1]
            idx_stop = i
            break
            
        times[i+1] = time.time() - t_start
            
    report = {}
    report['final_def']    = v_tmp
    report['energies_el']  = energies_el
    report['energies_ext'] = energies_ext
    report['residuals']    = residuals
    report['times']        = times
    report['idx_stop']     = idx_stop

    return report

def fd_validation_elastic(solid, v_def):
    torch.manual_seed(0)
    epsilons = torch.logspace(-9.0, -3.0, 100)
    perturb_global = 2.0e-3 * torch.rand(size=solid.v_rest.shape) - 1.0e-3
    v_def_perturb = solid.compute_pinned_deformation(v_def + perturb_global)
    perturb = 2.0 * torch.rand(size=solid.v_rest.shape) - 1.0
    perturb[solid.pin_idx] = 0.0
    
    # Back to original
    jac = solid.compute_jacobians(v_def_perturb)
    strain = solid.compute_strain_tensor(jac)
    f_el = solid.compute_elastic_forces(jac, strain)
    grad = -f_el
    grad[solid.pin_idx] = 0.0
    an_delta_E = (grad * perturb).sum()

    errors = []
    for eps in epsilons:
        # One step forward
        v_def_1 = v_def_perturb + eps * perturb
        jac_1 = solid.compute_jacobians(v_def_1)
        strain_1 = solid.compute_strain_tensor(jac_1)
        E1 = solid.compute_elastic_energy(jac_1, strain_1)

        # Two steps backward
        v_def_2 = v_def_perturb - eps * perturb
        jac_2 = solid.compute_jacobians(v_def_2)
        strain_2 = solid.compute_strain_tensor(jac_2)
        E2 = solid.compute_elastic_energy(jac_2, strain_2)

        fd_delta_E = (E1 - E2) / (2.0 * eps)    
        errors.append(abs(fd_delta_E - an_delta_E) / abs(an_delta_E))

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_ext(solid, v_def):
    torch.manual_seed(0)
    epsilons = torch.logspace(-9.0, -3.0, 100)
    perturb_global = 2.0e-3 * torch.rand(size=solid.v_rest.shape) - 1.0e-3
    v_def_perturb = solid.compute_pinned_deformation(v_def + perturb_global)
    perturb = 2.0 * torch.rand(size=solid.v_rest.shape) - 1.0
    perturb[solid.pin_idx] = 0.0
    
    # Back to original
    f_vol, f_ext = solid.compute_volumetric_and_external_forces()
    grad = -f_ext
    grad[solid.pin_idx] = 0.0
    an_delta_E = (grad * perturb).sum()
    
    errors = []
    for eps in epsilons:
        # One step forward
        v_def_1 = solid.compute_pinned_deformation(v_def_perturb + eps * perturb)
        def_barycenters_1 = compute_barycenters(v_def_1, solid.tet)
        E1 = solid.compute_external_energy(v_def_1, def_barycenters_1, f_vol)

        # One step backward
        v_def_2 = solid.compute_pinned_deformation(v_def_perturb - eps * perturb)
        def_barycenters_2 = compute_barycenters(v_def_2, solid.tet)
        E2 = solid.compute_external_energy(v_def_2, def_barycenters_2, f_vol)

        fd_delta_E = (E1 - E2) / (2.0 * eps)    
        errors.append(abs(fd_delta_E - an_delta_E) / abs(an_delta_E))
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()
    
def fd_validation_elastic_differentials(solid, v_def):
    torch.manual_seed(0)
    epsilons = torch.logspace(-9.0, -3.0, 100)
    perturb_global = 2.0e-3 * torch.rand(size=solid.v_rest.shape) - 1.0e-3
    v_def_perturb = solid.compute_pinned_deformation(v_def + perturb_global)
    perturb = 2.0 * torch.rand(size=solid.v_rest.shape) - 1.0
    perturb[solid.pin_idx] = 0.0

    jac = solid.compute_jacobians(v_def_perturb)
    strain = solid.compute_strain_tensor(jac)
    an_df = solid.compute_force_differentials(jac, strain, perturb)[solid.free_idx, :]
    an_df_full = torch.zeros(solid.v_rest.shape)
    an_df_full[solid.free_idx] = an_df.clone()
    errors = []

    for eps in epsilons:
        
        # One step forward
        v_def_1 = solid.compute_pinned_deformation(v_def_perturb + eps * perturb)
        jac_1 = solid.compute_jacobians(v_def_1)
        strain_1 = solid.compute_strain_tensor(jac_1)
        f1 = solid.compute_elastic_forces(jac_1, strain_1)[solid.free_idx, :]
        f1_full = torch.zeros(solid.v_rest.shape)
        f1_full[solid.free_idx] = f1

        # One step backward
        v_def_2 = solid.compute_pinned_deformation(v_def_perturb - eps * perturb)
        jac_2 = solid.compute_jacobians(v_def_2)
        strain_2 = solid.compute_strain_tensor(jac_2)
        f2 = solid.compute_elastic_forces(jac_2, strain_2)[solid.free_idx, :]
        f2_full = torch.zeros(solid.v_rest.shape)
        f2_full[solid.free_idx] = f2

        # Compute error
        fd_delta_f = (f1_full - f2_full) / (2.0 * eps)   
        norm_an_df = torch.linalg.norm(an_df_full)
        norm_error = torch.linalg.norm(an_df_full - fd_delta_f)
        errors.append(norm_error/norm_an_df)

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_inverse(optimizer):
    torch.manual_seed(0)
    epsilons = torch.logspace(-7.0, -1.0, 100)
    perturb_global = 1.0e-5 * (2.0 * torch.rand(size=optimizer.params.shape) - 1.0)
    optimizer.set_params(optimizer.params + perturb_global)
    optimizer.compute_gradient()
    params_save = optimizer.params.clone().detach()
    perturb = 2.0 * torch.rand(size=optimizer.params.shape) - 1.0
    grad = optimizer.compute_gradient()
    an_delta_E = (grad * perturb).sum()

    errors = []
    for eps in epsilons:
        # One step forward
        optimizer.set_params(params_save + perturb * eps)
        E1 = optimizer.compute_obj()

        # Two steps backward
        optimizer.set_params(params_save - perturb * eps)
        E2 = optimizer.compute_obj()

        # Compute error
        fd_delta_E = (E1 - E2) / (2.0 * eps)    
        errors.append(abs(fd_delta_E - an_delta_E) / abs(an_delta_E))

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

    
