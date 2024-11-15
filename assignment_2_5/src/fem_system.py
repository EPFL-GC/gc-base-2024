from conjugate_gradient import conjugate_gradient
import torch
from vertex_elt_sum import VertexEltsSum

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             FEM SYSTEM CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class FEMSystem():
    
    def __init__(self, v_rest, tet, ee, rho=1.0, pin_idx=None, f_mass=None):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            tet: indices of the element's vertices (#t, 4)
            ee: ElasticEnergy object that can be found in elastic_energy.py
            rho: mass per unit volume [kg.m-3]
            pin_idx: list of vertex indices to pin
            f_mass: torch tensor, external force per unit mass (3,) [N.kg-1]
        '''

        self.v_rest = v_rest.clone()
        self.tet = tet.clone()
        self.ee = ee
        self.rho = rho
        if pin_idx is None:
            pin_idx = []
        self.pin_idx = torch.tensor(pin_idx)
        self.free_idx = None
        self.free_mask = None
        self.make_free_indices_and_free_mask()
        self.f_mass = f_mass.clone()

        self.rest_barycenters = None
        self.Dm = None
        self.Bm = None
        self.W0 = None
        
        self.f_point = torch.zeros_like(v_rest)
        
        self.update_rest_shape(self.v_rest)
        self.vertex_tet_sum = VertexEltsSum(self.v_rest.shape[0], self.tet)
        
    def make_free_indices_and_free_mask(self):
        '''
        Updated attributes:
            free_idx: torch tensor of shape (#free_vertices,) containing the list of unpinned vertices
            free_mask: torch tensor of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''
        pass
        
    def update_rest_shape(self, v_rest):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
        
        Updated attributes:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            rest_barycenters: barycenters of the tetrahedra (#t, 3)
            Dm: shape matrices of the undeformed configuration (#t, 3, 3)
            Bm: inverse of the shape matrices of the undeformed configuration (#t, 3, 3)
            W0: signed volumes of the tetrahedra (#t,)
        '''
        pass
        
    def compute_pinned_deformation(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            v_def_pinned: deformed position of the vertices of the mesh after taking pinning into account (#v, 3)
        '''
        pass
        
    def compute_jacobians(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            jac: Jacobians of the deformation (#t, 3, 3)
        '''
        pass
    
    def compute_strain_tensor(self, jac):
        '''
        Args:
            jac: Jacobians of the deformation (#t, 3, 3)

        Returns:
            E: strain induced by the deformation (#t, 3, 3)
        '''
        pass
    
    ## Energies ##
    
    def compute_elastic_energy(self, jac, E):
        '''
        Args:
            jac: Jacobians of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3)
        
        Returns:
            energy_el: elastic energy of the system [J]
        '''
        pass
    
    def compute_external_energy(self, v_def, def_barycenters, f_vol):
        '''
        This computes the external energy potential
        
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)
            def_barycenters: barycenters of the deformed tetrahedra (#t, 3)
            f_vol: external force per unit volume acting on the tets (#t, 3)

        Returns:
            energy_ext : postential energy due to external forces [J]
        '''

        pass
    
    ## Forces ##
    
    def compute_elastic_forces(self, jac, E):
        '''
        Args:
            jac: the jacobian of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3), can be None for Neo-Hookean
            
        Returns:
            forces_el: the elastic forces of the system (#v, 3)
        '''
        pass
    
    def compute_volumetric_and_external_forces(self):
        '''
        Convert force per unit mass to volumetric forces (in the undeformed state), then distribute
        the forces to the vertices of the mesh.

        Returns:
            f_vol: torch tensor of shape (#t, 3) external force per unit volume acting on the tets
            f_ext: torch tensor of shape (#v, 3) external force acting on the vertices
        '''
        pass

    def add_point_load(self, v_idx, f_point):
        '''
        Args:
            v_idx: tensor of size (n_idx,) containing the list of vertex ids on which force is applied
            f_point: tensor of size (n_idx, 3) containing force to add at each vertex
        
        Updated attributes:
            f_point : tensor of size (#v, 3)
        '''
        pass
    
    ## Force Differentials ##
    
    def compute_force_differentials(self, jac, E, v_disp):
        '''
        This computes the differential of the force given a displacement dx,
        where df = df/dx|x . dx = - K(x).dx. The matrix vector product K(x)w
        is then given by the call self.compute_force_differentials(-w).

        Args:
            jac: the jacobian of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3), can be None for Neo-Hookean
            v_disp: displacement of the vertices of the mesh (#v, 3)

        Returns:
            dforces_el: force differentials at the vertices of the mesh (#v, 3)
        ''' 
        pass
    
    ## Equilibrium ##
    
    def equilibrium_step(self, v, jac, E, step_size_init=2.0, max_l_iter=20, c1=1.0e-4, verbose=False):
        '''
        This function computes the next vertex positions of the whole solid
        taking a Newton-CG step.

        Args:
            v: current position of the vertices of the mesh (#v, 3)
            jac: the current jacobian of the deformation (#t, 3, 3)
            E: strain induced by the current deformation (#t, 3, 3), can be None for Neo-Hookean
            step_size_init: initial step size for the line search
            max_l_iter: maximum number of line search iterations
            c1: Armijo condition parameter
            verbose: if True, prints the energy and force residual norm at each line search iteration
            
        Returns:
            v_new: position of the vertices of the mesh after the step (#v, 3)
            jac_new: jacobian of the deformation after the step (#t, 3, 3)
            E_new: strain induced by the deformation after the step (#t, 3, 3)
        '''

        ## TODO: Compute initial forces
        ## f_vol, f_ext are the volumetric and external forces
        ## f_el is the elastic forces
        ## ft is the total forces
        f_vol, f_ext = ...
        f_el = ...
        ft  = ...
        
        ## TODO: Define LHS
        def LHS(dx):
            '''
            Should implement the Hessian-Vector Product L(dx), and take care of pinning constraints
            as described in the handout.
            '''
            return None
        self.LHS = LHS # Save to class for testing
        
        ## TODO: Define RHS
        RHS = ...
        
        self.RHS = RHS # Save to class for testing

        ## TODO: Use conjugate gradient to find a descent direction
        ## dx0 has shape (#v, 3)
        ## (see conjugate_gradient in conjugate_gradient.py)
        dx0 = ...

        dx_CG = dx0.clone()
        
        # Run line search on the direction
        step_size = step_size_init
        def_barycenters = compute_barycenters(v, self.tet)
        ## TODO: Compute initial energy
        ## energy_tot_prev is the current total energy and serves as a reference for the armijo rule
        energy_tot_prev = ...
        for l_iter in range(max_l_iter):
            step_size *= 0.5
            v_search = self.compute_pinned_deformation(v + dx_CG * step_size)
            jac_search = self.compute_jacobians(v_search)
            E_search = self.compute_strain_tensor(jac_search)
            def_barycenters_search = compute_barycenters(v_search, self.tet)

            ## TODO: Check if the armijo rule is satisfied
            ## energy_tot_search is the current total energy
            ## armijo is a boolean that says whether the condition is satisfied
            energy_tot_search = ...
            armijo            = ...
            
            if armijo or l_iter == max_l_iter-1:
                if verbose:
                    ## TODO: Compute final forces
                    ## f_vol_new, f_ext_new are the volumetric and external forces
                    ## f_el_new is the elastic forces
                    ## ft_new is the total forces
                    f_vol_new, f_ext_new = ...
                    f_el_new = ...
                    ft_new  = ...
                    g_new = torch.linalg.norm(ft_new[self.free_idx, :])
                    print(f"Energy: {energy_tot_search.item():.3e}. Force residual norm: {g_new:.3e}. Line search Iters: {l_iter}")
                break
            
        return v_search, jac_search, E_search
    
    def find_equilibrium(self, v_init, n_steps=1000, thresh=1.0, verbose=False):
        '''
        Args:
            v_init: initial guess for the equilibrium position (#v, 3)
            n_steps: maximum number of optimization steps
            thresh: threshold on the force value [N]
            verbose: if True, prints the energy and force residual norm at each iteration
            
        Returns:
            v_tmp: position of the vertices of the mesh at equilibrium (#v, 3)
        '''
        
        v_tmp = v_init.clone()
        jac_tmp = self.compute_jacobians(v_tmp)
        E_tmp = self.compute_strain_tensor(jac_tmp)
        _, f_ext = self.compute_volumetric_and_external_forces()
        
        for _ in range(n_steps):
            # Take a Newton-CG step
            v_tmp, jac_tmp, E_tmp = self.equilibrium_step(v_tmp, jac_tmp, E_tmp, verbose=verbose)

            # Measure the force residuals
            f_el_tmp = self.compute_elastic_forces(jac_tmp, E_tmp)
            ft_tmp  = f_ext + f_el_tmp
            residuals_tmp = torch.linalg.norm(ft_tmp[self.free_idx, :])
        
            if residuals_tmp < thresh:
                break
        if verbose: print("Final residuals (equilibrium): {:.2e}".format(to_numpy(residuals_tmp)))
        
        return v_tmp
    
def compute_barycenters(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)
        
    Returns:
        barycenters: barycenters of the tetrahedra
    '''
    pass

def compute_shape_matrices(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)

    Returns:
        D: shape matrices of current configuration (#t, 3, 3)
    '''
    pass

def compute_signed_volume(D):
    '''
    Args:
        D: shape matrices of current configuration (#t, 3, 3)

    Returns:
        signed_volume: signed volumes of the tetrahedra (#t,)
    '''
    pass
