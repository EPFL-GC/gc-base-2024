import numpy as np

def create_vol_cube(nX, nY, nZ):
    '''
    Args:
        nX: number of vertices along the x axis
        nY: number of vertices along the y axis
        nZ: number of vertices along the z axis
        
    Returns:
        v: tensor of shape (#v, 3) containing the vertices of the cube
        t: tensor of shape (#t, 4) containing the tetrahedra of the cube
    '''

    v = np.zeros(shape=(nX, nY, nZ, 3))
    v[..., 0] = v[..., 0] + np.linspace(0.0, 1.0, nX).reshape(-1, 1, 1)
    v[..., 1] = v[..., 1] + np.linspace(0.0, 1.0, nY).reshape(1, -1, 1)
    v[..., 2] = v[..., 2] + np.linspace(0.0, 1.0, nZ).reshape(1, 1, -1)
    v = v.reshape(-1, 3)

    trisFace1 = np.array([[idY * nZ + idZ, (idY + 1) * nZ + idZ + 1, idY * nZ + idZ + 1] for idZ in range(nZ-1) for idY in range(nY-1)])
    trisFace2 = np.array([[idY * nZ + idZ, (idY + 1) * nZ + idZ, (idY + 1) * nZ + idZ + 1] for idZ in range(nZ-1) for idY in range(nY-1)])

    nPtsPerLayer = nY * nZ
    tetsTmp = np.empty(shape=(0, 4)).astype(np.int64)
    trisPrev1 = trisFace1.copy()
    trisPrev2 = trisFace2.copy()
    for idLayer in range(nX-1):
        trisNew1 = (idLayer + 1) * nPtsPerLayer + trisFace1.copy()
        trisNew2 = (idLayer + 1) * nPtsPerLayer + trisFace2.copy()
        tetsNew = np.concatenate([
            np.concatenate([trisPrev1, trisNew1[:, 2].reshape(-1, 1)], axis=1),
            np.concatenate([trisPrev2, trisNew1[:, 2].reshape(-1, 1)], axis=1),
            np.concatenate([trisPrev2[:, :2], trisNew1[:, 2].reshape(-1, 1), trisNew1[:, 0].reshape(-1, 1)], axis=1),
            np.concatenate([trisPrev2[:, 1].reshape(-1, 1), trisNew1], axis=1),
            np.concatenate([trisPrev2[:, 1].reshape(-1, 1), trisNew2], axis=1),
            np.concatenate([trisPrev2[:, 1:], trisNew1[:, 2].reshape(-1, 1), trisNew1[:, 1].reshape(-1, 1)], axis=1),
        ], axis=0)
        tetsTmp = np.concatenate([tetsTmp, tetsNew], axis=0)
        trisPrev1 = trisNew1.copy()
        trisPrev2 = trisNew2.copy()

        
    D = (v[tetsTmp[:, :-1]] - v[tetsTmp[:, 3]][:, None]).transpose(0, 2, 1)
    signedVolume = - np.linalg.det(D) / 6.0
    isFlipped = signedVolume < 0.0
    tets0 = tetsTmp[isFlipped, 0].copy()
    tetsTmp[isFlipped, 0] = tetsTmp[isFlipped, 1]
    tetsTmp[isFlipped, 1] = tets0
    
    return v, tetsTmp

