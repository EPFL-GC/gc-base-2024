import torch

def conjugate_gradient(L_method, b):
    '''
    Finds an inexact Newton descent direction using Conjugate Gradient (CG)
    Solves partially L(x) = b where A is positive definite, using CG.
    The method should be implemented to check whether the added direction is
    an ascent direction or not, and whether the residuals are small enough.
    Details can be found in the handout.

    Args:
        L_method: a method that computes the Hessian vector product. It should
                    take an tensor of shape (n,) and return an tensor of shape (n,)
        b: right hand side of the linear system (n,)

    Returns:
        p_star: torch tensor of shape (n,) solving the linear system approximately
    '''

    return torch.zeros_like(b)