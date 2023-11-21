from abc import abstractmethod
from typing import Dict, List, Union, Tuple
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from numpy.random import randint
import sys

#sys.path.append('/DIG/')

from utils.drapenet_structure import DIGStructure
from utils.mesh_utils import concatenate_meshes
from utils.colors import (
    GarmentColors, 
    BodyColors,
    norm_color
)


def random_pallete_color(pallete):
    return np.array(norm_color(list(pallete)[randint(0, len(pallete) - 1)].value))


def default_body_color(pallete):
    return np.array(norm_color(list(pallete)[0].value))


def default_upper_color(pallete):
    return np.array(norm_color(list(pallete)[4].value))


def default_lower_color(pallete):
    return np.array(norm_color(list(pallete)[7].value))


class MeshManager(object):

    """
    An abstract mesh manager with common declaration/definitions.
    """

    @abstractmethod
    def create_meshes(
        self
    ):
        """
        Implemented for various types for different mesh manager subclasses.
        """
        pass

    def save_meshes(
            self,
            meshes,
            save_basepath: str
    ) -> None:
        """
        Simply save the provided body and garment meshes as obj.
        """
        meshes[0].write_obj(f'{save_basepath}-body.obj')
        meshes[1].write_obj(f'{save_basepath}-upper.obj')
        meshes[2].write_obj(f'{save_basepath}-lower.obj')


class ColoredGarmentsMeshManager(MeshManager):

    ''' The mesh manager for colored garment meshes.
    
        This class is used for rendering clothed meshes in a numpy environment,
        such as when generating the training data.
    '''

    def __init__(self):
        super().__init__()

    def create_meshes(
            self,
            garment_output_dict,
            device: str = 'cpu'
    ) -> List[Meshes]:
        ''' Extract trimesh Meshes from SMPL4Garment output (verts and faces).
        
            To construct the Meshes for upper, lower, and both piece of
            clothing on top of the body mesh, the vertices and the faces need
            to be concatenated into corresponding arrays. In particular, the
            body mesh only consists of body vertices and body faces, i.e.,
            is not concatenated with other arrays. The lower+body garment 
            mesh consists of concatenated body and lower mesh vertices and 
            faces. Finally, the complete mesh (body+upper+lower) consists of
            concatenanted body, lower, and upper vertices and faces. The
            three meshes are returnned as a result of this method.
        '''
        verts_list = [
            garment_output_dict['upper'].body_verts,
            garment_output_dict['upper'].garment_verts,
            garment_output_dict['lower'].garment_verts
        ]
        faces_list = [
            garment_output_dict['upper'].body_faces,
            garment_output_dict['upper'].garment_faces,
            garment_output_dict['lower'].garment_faces
        ]
        
        body_colors = np.ones_like(verts_list[0]) * \
            default_body_color(BodyColors)
        
        part_colors_list = [
            np.ones_like(verts_list[1]) * default_upper_color(GarmentColors),
            np.ones_like(verts_list[2]) * default_lower_color(GarmentColors),
        ]
        
        concat_verts_list = [verts_list[0]]
        concat_faces_list = [faces_list[0]]
        concat_color_list = [body_colors]
        for idx in range(len(verts_list)-1):
            concat_verts, concat_faces = concatenate_meshes(
                vertices_list=[concat_verts_list[idx], verts_list[idx+1]],
                faces_list=[concat_faces_list[idx], faces_list[idx+1]]
            )
            concat_verts_list.append(concat_verts)
            concat_faces_list.append(concat_faces)
            
            part_colors = part_colors_list[idx]
            
            concat_color_list.append(
                np.concatenate([concat_color_list[idx], part_colors], axis=0))
        
        meshes = []
        for idx in range(len(verts_list)):
            concat_verts_list[idx] = torch.from_numpy(
                concat_verts_list[idx]).float().unsqueeze(0).to(device)
            concat_faces_list[idx] = torch.from_numpy(
                concat_faces_list[idx].astype(np.int32)).unsqueeze(0).to(device)
            concat_color_list[idx] = torch.from_numpy(
                concat_color_list[idx]).float().unsqueeze(0).to(device)
            
            meshes.append(Meshes(
                verts=concat_verts_list[idx],
                faces=concat_faces_list[idx],
                textures=Textures(verts_rgb=concat_color_list[idx])
            ))
        
        return meshes 
