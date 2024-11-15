import meshplot as mp
import torch

shadingOptions = {
    "flat":True,
    "wireframe":False,   
}

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

def get_plot(solid, v_def, be, rot, length_scale, target_mesh=None, be_target=None):
    _, f_ext = solid.compute_volumetric_and_external_forces()
    p = mp.plot(to_numpy(v_def @ rot.T), to_numpy(solid.tet), shading=shadingOptions)
    p.add_points(to_numpy(v_def[solid.pin_idx, :] @ rot.T), shading={"point_color":"black", "point_size": 0.1 * length_scale})
    forcesScale = 2.0 * torch.max(torch.linalg.norm(f_ext, axis=1))
    p.add_lines(to_numpy(v_def @ rot.T), to_numpy((v_def + length_scale * f_ext / forcesScale) @ rot.T))
    p.add_edges(to_numpy(solid.v_rest @ rot.T), be, shading={"line_color": "blue"})
    if not target_mesh is None:
        p.add_edges(to_numpy(target_mesh @ rot.T), be_target, shading={"line_color": "red"})

    return p

def plot_torch_solid(solid, v_def, be, rot, length_scale, iteration = None, target_mesh=None, be_target=None, plot = None, rest_color = "blue", target_color = "red"):
    '''
    Args:
        solid: fem system to visualize
        v_def: deformed vertices, torch tensor of shape (#v, 3)
        be: boundary edges
        rot: transformation matrix to apply (here we assume it is a rotation)
        length_scale: length scale of the mesh, used to represent pinned vertices
        target_mesh: target mesh to compare with (#v, 3)
        be_target: target boundary edges
    '''
    _, f_ext = solid.compute_volumetric_and_external_forces()
    if (plot is None):
        p = mp.plot(to_numpy(v_def @ rot.T), to_numpy(solid.tet), shading=shadingOptions, plot = plot)
        p.add_points(to_numpy(v_def[solid.pin_idx, :] @ rot.T), shading={"point_color":"black", "point_size": 0.1 * length_scale})
        forcesScale = 2.0 * torch.max(torch.linalg.norm(f_ext, axis=1))
        p.add_lines(to_numpy(v_def @ rot.T), to_numpy((v_def + length_scale * f_ext / forcesScale) @ rot.T))
        p.add_edges(to_numpy(solid.v_rest @ rot.T), be, shading={"line_color": rest_color})
        if not target_mesh is None:
            p.add_edges(to_numpy(target_mesh @ rot.T), be_target, shading={"line_color": target_color})
    else:
        plot.update_object(vertices=to_numpy(v_def @ rot.T))
        forcesScale = 2.0 * torch.max(torch.linalg.norm(f_ext, axis=1))
        plot.add_lines(to_numpy(v_def @ rot.T), to_numpy((v_def + length_scale * f_ext / forcesScale) @ rot.T))
        plot.add_edges(to_numpy(solid.v_rest @ rot.T), be, shading={"line_color": rest_color})
        print("rest_state", to_numpy(solid.v_rest @ rot.T)[0], to_numpy(solid.v_rest @ rot.T)[10], to_numpy(solid.v_rest @ rot.T)[-2])
        if iteration == 0:
            plot.remove_object(2)
            # plot.remove_object(3)
        else:
            plot.remove_object(5 + 2 * (iteration - 1))
            # plot.remove_object(6 + 2 * (iteration - 1))