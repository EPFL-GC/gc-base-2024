import torch

class VertexEltsSum:
    
    def __init__(self, nv, elts):
        '''
        Args:
            nv: number of vertices
            elts: torch tensor of shape (#e, 3 or 4), containing the indices of the vertices of each element of the mesh
        '''
        self.i  = elts.T.flatten()
        self.j  = torch.tile(torch.arange(elts.shape[0]), (elts.shape[1],)).to(elts.device)
        self.indices = torch.stack([self.i, self.j], dim=0)
        self.nv = nv
        self.ne = elts.shape[0]
        self.vert_per_elt = elts.shape[1]
        
    def vertex_elt_sum(self, data):
        '''
        Distributes data specified at each element to the neighboring vertices.
        All neighboring vertices will receive the value indicated at the corresponding tet position in data.

        Args:
            data: torch tensor of shape (4 * #e,), flattened in a column-major fashion

        Returns:
            data_sum: torch array of shape (#v,), containing the summed data
        '''
        v_sum = torch.sparse_coo_tensor(self.indices, data, (self.nv, self.ne))
        return torch.sparse.mm(v_sum, torch.ones(size=(self.ne, 1))).flatten()
