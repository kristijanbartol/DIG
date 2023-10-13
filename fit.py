import argparse
import numpy as np
import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from networks import (
    IGR,
    lbs_mlp,
    learnt_representations
)
from rendering.renderer import Renderer
from smpl_pytorch.smpl_server import SMPLServer
from utils.deform import (
    deform,
    reconstruct,
    rotate_root_pose_x
)
from utils.loaders import (
    load_poseshape,
    load_segmaps
)


D_WIDTH = 512
DIM_THETA = 72
DIM_THETA_P = 128 
DIM_LATENT_G = 12
NUM_G = 100


def get_mesh_sdf(
        verts,  
        style, 
        model_G
    ):

    ### 2: evaluate analytical normals
    verts_torch = torch.from_numpy(verts).float().cuda()
    verts_torch.requires_grad = True
    num_points = len(verts_torch)
    x_cloth_points = style.unsqueeze(0).repeat(num_points, 1)
    pred_sdf_verts = model_G(verts_torch, x_cloth_points, num_points)
    pred_sdf_verts.sum().backward(retain_graph=True)

    normals = verts_torch.grad
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 0.0001)

    ### 3: new verts= old vert + sdf (=0) * normals, to bring bck differentiability
    new_verts = verts_torch.detach() - pred_sdf_verts * normals

    return new_verts


def update_z_shirt(
        model_G,
        pose, 
        beta, 
        z_style, 
        images_gt, 
        tfs_c_inv, 
        shapedirs, 
        tfs_weighted_zero, 
        embedder, 
        model_lbs, 
        model_lbs_delta, 
        model_blend_weight, 
        renderer, 
        device,
        iters=100
    ):

    lr = 1e-2
    z_style.requires_grad = True 
    pose.requires_grad = False 
    beta.requires_grad = False 
    optimizer = torch.optim.Adam([{'params': z_style, 'lr': lr}])

    with torch.no_grad():
        smpl_verts_pred, joints_pred, _, _, smpl_tfs = smpl_server.forward_verts(betas=beta,
                                                transl=np.zeros(3,),
                                                body_pose=pose[:, 3:],
                                                global_orient=pose[:, :3],
                                                return_verts=True,
                                                return_full_pose=True,
                                                v_template=smpl_server.v_template, rectify_root=False)
        smpl_verts_pred = smpl_verts_pred.squeeze()
        smpl_tfs = smpl_tfs.squeeze()
        
        smpl_verts_pred.requires_grad = False 
        smpl_tfs.requires_grad = False 

    smpl_faces = smpl_server.smpl.faces

    loss_min = 1e10
    z_style_best = z_style.clone().detach()
    for step in range(iters):

        with torch.no_grad():
            cloth_mesh = reconstruct(z_style, model_G, just_vf=False, resolution=256)
            gar_verts, gar_faces = cloth_mesh.vertices, cloth_mesh.faces 

        verts_zero = torch.zeros(len(smpl_verts_pred)+len(gar_verts), 3)
        faces = torch.cat((smpl_faces.cpu(), torch.LongTensor(gar_faces) + len(smpl_verts_pred)))
        # assign labels for body and cloth
        smpl_rgb = torch.zeros(len(smpl_verts_pred), 3)
        smpl_rgb[:,0] += 255
        gar_rgb = torch.zeros(len(gar_verts), 3)
        gar_rgb[:,1] += 255
        verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        mesh = Meshes(
            verts=[verts_zero.to(device)],   
            faces=[faces.to(device)],
            textures=textures
        )
        new_vertices = get_mesh_sdf(gar_verts, z_style, model_G)

        verts_gar_deformed, _, _, _ = deform(
            new_vertices, 
            gar_faces, 
            smpl_tfs, 
            tfs_c_inv, 
            pose, 
            beta, 
            shapedirs, 
            tfs_weighted_zero, 
            embedder, 
            model_lbs, 
            model_lbs_delta, 
            model_blend_weight
        )

        verts_deformed = torch.cat((smpl_verts_pred, verts_gar_deformed), dim=0)
        # this is just for coordinate matching since SMPL and py3d have different definition for the world coordinate.
        signs = torch.ones_like(verts_deformed).cuda()
        signs[:,:2] *= -1
        verts_deformed *= signs

        new_src_mesh = mesh.offset_verts(verts_deformed)
        images_predicted = renderer(new_src_mesh)
        images_pred = images_predicted[0, :, :, :3]/255
        
        gar_silh = images_pred[:, :, 1]
        gar_gt = images_gt[:, :, 1]
        intersection_g = (gar_silh*gar_gt).sum()
        union_g = gar_silh.sum() + gar_gt.sum() - intersection_g
        loss_texture = (1 - intersection_g/union_g)*224#10
        loss_reg = z_style.norm()/10

        line = 'step: %3d, loss_texture: %0.4f, loss_reg: %0.5f'%(step, loss_texture.item(), loss_reg.item())
        print(line)

        loss = loss_texture + loss_reg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return z_style_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='demo/images')
    parser.add_argument('--labels_folder', type=str, default='demo/npz')
    parser.add_argument('--masks_folder', type=str, default='demo/masks')
    parser.add_argument('--output_folder', type=str, default='output/')
    args = parser.parse_args()

    ''' Load pretrained models and necessary files '''
    data = np.load('extra-data/shapedirs_f.npz')
    shapedirs = torch.FloatTensor(data['shapedirs']).cuda()
    tfs_weighted_zero = torch.FloatTensor(data['tfs_weighted_zero']).cuda()
    lbs_weights = torch.FloatTensor(data['lbs_weights']).cuda()

    model_lbs = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=D_WIDTH, depth=8, skip_layer=[4])
    model_lbs.load_state_dict(torch.load('extra-data/pretrained/lbs_shirt.pth'))
    model_lbs = model_lbs.cuda().eval()
    model_lbs_p = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=D_WIDTH, depth=8, skip_layer=[4])
    model_lbs_p.load_state_dict(torch.load('extra-data/pretrained/lbs_pants.pth'))
    model_lbs_p = model_lbs_p.cuda().eval()

    embedder, embed_dim = lbs_mlp.get_embedder(4)

    model_lbs_delta = lbs_mlp.lbs_pbs(
        d_in_theta=DIM_THETA, 
        d_in_x=embed_dim, 
        d_out_p=DIM_THETA_P, 
        skip=True, 
        hidden_theta=D_WIDTH, 
        hidden_matrix=D_WIDTH
    )
    model_lbs_delta = model_lbs_delta.cuda().eval()
    model_lbs_delta.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_shirt.pth'))
    model_lbs_delta_p = lbs_mlp.lbs_pbs(
        d_in_theta=DIM_THETA, 
        d_in_x=embed_dim, 
        d_out_p=DIM_THETA_P, 
        skip=True, 
        hidden_theta=D_WIDTH, 
        hidden_matrix=D_WIDTH
    )
    model_lbs_delta_p = model_lbs_delta_p.cuda().eval()
    model_lbs_delta_p.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_pants.pth'))

    model_blend_weight = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=len(shapedirs), width=512, depth=8, skip_layer=[4]).cuda().eval()
    model_blend_weight.load_state_dict(torch.load('extra-data/pretrained/blend_weight.pth'))

    ''' Initialize SMPL model '''
    rest_pose = np.zeros((24,3), np.float32)
    rest_pose[1,2] = 0.15
    rest_pose[2,2] = -0.15
    rest_pose = rotate_root_pose_x(rest_pose)
    param_canonical = torch.zeros((1, 86),dtype=torch.float32).cuda()
    param_canonical[0, 0] = 1
    param_canonical[:,4:76] = torch.FloatTensor(rest_pose).reshape(-1)

    smpl_server = SMPLServer(
        param_canonical, 
        gender='f', 
        betas=None, 
        v_template=None
    )
    tfs_c_inv = smpl_server.tfs_c_inv.detach()
    renderer = Renderer()

    img_names = os.listdir(args.img_folder)

    for img_name in img_names:
        params_dict = load_params_dict(img_name)
        seg_maps_dict = load_segmaps_dict(img_name)
        for garment_idx, garment_part in enumerate(['shirt', 'pants']):
            model_G = IGR.ImplicitNet_multiG(d_in=3+DIM_LATENT_G, skip_in=[4]).cuda().eval()
            model_G.load_state_dict(torch.load(f'extra-data/pretrained/{garment_part}.pth'))
            model_G = model_G.cuda().eval()

            model_rep = learnt_representations.Network(cloth_rep_size=DIM_LATENT_G, samples=NUM_G)
            model_rep.load_state_dict(torch.load(f'extra-data/pretrained/{garment_part}_rep.pth'))
            model_rep = model_rep.cuda().eval()

            init_z_style = model_rep.weights[0]

            z_style_best = update_z_shirt(
                model_G,
                pose, 
                beta, 
                init_z_style, 
                seg_maps[garment_idx], 
                tfs_c_inv, 
                shapedirs, 
                tfs_weighted_zero, 
                embedder, 
                model_lbs, 
                model_lbs_delta, 
                model_blend_weight, 
                renderer,
                device='cuda:0', 
                iters=100
            )
