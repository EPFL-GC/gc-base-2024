import torch

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Args:
            young: Young's modulus [Pa]
            poisson: Poisson ratio
        '''
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.mu = young / (2.0 * (1.0 + poisson))
        
    def compute_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
        
        Returns:
            E: strain induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError
    
    def compute_differential_strain_tensor(self, jac, dJac):
        '''
        This method computes the differential of strain tensor (#e, 3, 3)

        Args:
            jac: jacobian of the deformation (#e, 3, 3)
            dJac: differential of the jacobian of the deformation (#e, 3, 3)
            
        Returns:
            dE: differential of the strain tensor (#e, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_energy_density(self, jac, E):
        '''
        This method computes the energy density at each tetrahedron (#t,)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
            E: strain tensor per element (#e, 3, 3)
        
        Returns:
            psi: energy density per tet (#t,)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        '''
        This method computes the stress tensor (#t, 3, 3)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
            E: strain tensor per element (#e, 3, 3)
        
        Returns:
            P: stress tensor induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError
    
    def compute_differential_piola_kirchhoff_stress_tensor(self, jac, dJac, E, dE):
        '''
        This method computes the differential of the stress tensor (#e, 3, 3)

        Args:
            jac: jacobian of the deformation (#e, 3, 3)
            dJac: differential of the jacobian of the deformation (#e, 3, 3)
            E: strain tensor per element (#e, 3, 3)
            dE: differential of the strain tensor per element (#e, 3, 3)
            
        Returns:
            dP: differential of the stress tensor per element (#e, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
    
    def compute_strain_tensor(self, jac):
        pass
    
    def compute_differential_strain_tensor(self, jac, dJac):
        pass

    def compute_energy_density(self, jac, E):
        pass
    
    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        pass

    def compute_differential_piola_kirchhoff_stress_tensor(self, jac, dJac, E, dE):
        pass


class NeoHookeanElasticEnergy(ElasticEnergy):
    '''
    Implements the Neo-Hookean energy density function \psi_E in https://graphics.pixar.com/library/StableElasticity/paper.pdf.
    That elasticity model does not have singularities for degenerate deformations, is reflection stable, and remains rest stable.
    '''
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.muNH = self.mu
        self.lbdaNH = self.lbda + self.muNH
        self.nuNH = (self.lbdaNH - self.muNH) / (2.0 * self.lbdaNH)
        self.alpha = 0.0 # TODO: compute alpha so that the model is rest-stable
        self.psi0 = 0.0 # TODO: compute psi0 so that the model has 0 energy at rest

    def compute_strain_tensor(self, jac):
        pass
    
    def compute_differential_strain_tensor(self, jac, dJac):
        pass

    def compute_energy_density(self, jac, E):
        pass

    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        pass

    def compute_differential_piola_kirchhoff_stress_tensor(self, jac, dJac, E, dE):
        pass
