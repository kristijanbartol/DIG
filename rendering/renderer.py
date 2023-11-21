from typing import Dict, Tuple, Optional, List
import torch.nn as nn
import torch
import numpy as np

from configs.const import (
    CAM_DIST,
    FEATURE_SIZE,
    MEAN_CAM_T,
    MEAN_CAM_Y_OFFSET,
    LIGHT_T,
    LIGHT_AMBIENT_COLOR,
    LIGHT_DIFFUSE_COLOR,
    LIGHT_SPECULAR_COLOR,
    BACKGROUND_COLOR,
    ORTHOGRAPHIC_SCALE
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    BlendParams
)

from rendering.mesh_manager import ColoredGarmentsMeshManager


class Renderer(nn.Module):

    default_cam_R, default_cam_t = look_at_view_transform(
        dist=CAM_DIST,
        #dist=-2.7,
        elev=0,
        azim=0,
        degrees=True
    )
    default_cam_t[:, 1] += MEAN_CAM_Y_OFFSET

    assert(default_cam_t[:, 0] == MEAN_CAM_T[0])
    assert(default_cam_t[:, 1] == MEAN_CAM_T[1])
    assert(default_cam_t[:, 2] == MEAN_CAM_T[2])

    def __init__(
            self,
            device: str,
            cam_t: Optional[torch.Tensor] = None,
            cam_R: Optional[torch.Tensor] = None,
            img_wh: int = 256,
            projection_type: str = 'perspective',
            orthographic_scale: float = ORTHOGRAPHIC_SCALE,
            blur_radius: float = 0.0,
            faces_per_pixel: int = 1,
            bin_size: int = None,
            max_faces_per_bin: int = None,
            perspective_correct: bool = False,
            cull_backfaces: bool = False,
            clip_barycentric_coords: bool = None,
            light_t: Tuple[float] = LIGHT_T,
            light_ambient_color: Tuple[float] = LIGHT_AMBIENT_COLOR,
            light_diffuse_color: Tuple[float] = LIGHT_DIFFUSE_COLOR,
            light_specular_color: Tuple[float] = LIGHT_SPECULAR_COLOR,
            background_color: Tuple[float] = BACKGROUND_COLOR
        ) -> None:
        super().__init__()
        self.img_wh = FEATURE_SIZE
        self.device = device

        # Cameras - pre-defined here but can be specified in forward 
        # pass if cameras will vary (e.g. random cameras).
        if projection_type not in ['perspective', 'orthographic']: 
            print('Invalid projection type:', projection_type)
            print('Setting to: perspective')
            projection_type = 'perspective'
        print('\nRenderer projection type:', projection_type)
        self.projection_type = projection_type

        cam_R = self.default_cam_R if cam_R is None else cam_R
        cam_t = self.default_cam_t if cam_t is None else cam_t
        
        self.cameras = FoVOrthographicCameras(
            device=device,
            R=cam_R,
            T=cam_t,
            scale_xyz=((
                orthographic_scale,
                orthographic_scale,
                1.5),)
        )

        # Lights for textured RGB render - pre-defined here but can be specified in 
        # forward pass if lights will vary (e.g. random cameras).
        self.lights_rgb_render = PointLights(
            device=device,
            location=light_t,
            ambient_color=light_ambient_color,
            diffuse_color=light_diffuse_color,
            specular_color=light_specular_color
        )

        # Rasterizer
        raster_settings = RasterizationSettings(
            image_size=img_wh,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
            perspective_correct=perspective_correct,
            cull_backfaces=cull_backfaces,
            clip_barycentric_coords=clip_barycentric_coords
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, 
            raster_settings=raster_settings
        )  # Specify camera in forward pass

        # Shader for textured RGB output and IUV output
        self.blend_params = BlendParams(background_color=background_color)
        self.rgb_shader = HardPhongShader(
            device=device, 
            cameras=self.cameras,
            lights=self.lights_rgb_render, 
            blend_params=self.blend_params
        )
        self.to(device)

    def to(self, device):
        '''Move tensors to specified device.'''
        self.rasterizer.to(device)
        self.rgb_shader.to(device)

    def _update_lights_settings(
            self, 
            new_lights_settings: Dict
        ) -> None:
        '''Update lights settings by directly setting PointLights properties.'''
        self.lights_rgb_render.location = new_lights_settings['location']
        self.lights_rgb_render.ambient_color = new_lights_settings['ambient_color']
        self.lights_rgb_render.diffuse_color = new_lights_settings['diffuse_color']
        self.lights_rgb_render.specular_color = new_lights_settings['specular_color']

    def _process_optional_arguments(
            self,
            cam_t: Optional[np.ndarray] = None,
            orthographic_scale: Optional[float] = None,
            lights_rgb_settings: Optional[Dict[str, Tuple[float]]] = None
        ) -> None:
        '''Update camera translation, focal length, or lights settings.'''
        if cam_t is not None:
            self.cameras.T = torch.from_numpy(cam_t).float().unsqueeze(0).to(self.device)
        if orthographic_scale is not None and self.projection_type == 'orthographic':
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)
        if lights_rgb_settings is not None and self.render_rgb:
            self._update_lights_settings(lights_rgb_settings)
    


class DIGRenderer(Renderer):

    ''' Clothed meshes renderer.
    
        Note that the class is used to render ClothSURREAL examples.
        Also note that the implementation is Numpy-based, because
        it will not happen that the tensors will arrive as params.
        This is because the current parametric model, TailorNet,
        is used only offline to generate input data and not during
        training.
    '''

    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        ''' The clothed renderer constructor.'''
        super().__init__(*args, **kwargs)
        self.mesh_manager = ColoredGarmentsMeshManager()

    def _extract_seg_maps(
            self, 
            rgbs: List[np.ndarray]
        ) -> np.ndarray:
        ''' Extract segmentation maps from the RGB renders of meshes.

            Note there is a specific algorithm in this procedure. First, take
            the whole clothed body image and use it to create the first map.
            Then, use RGB image with one less piece of garment to get the 
            difference between this image and the previous one. The difference
            is exactly the segmentation map of this piece of garment. Finally,
            apply this procedure for the second piece of clothing.
        '''
        maps = []
        rgb = np.zeros_like(rgbs[-1][0])
        for rgb_idx in range(len(rgbs) - 1, -1, -1):
            seg_map = ~np.all(np.isclose(rgb, rgbs[rgb_idx], atol=1e-3), axis=-1)
            maps.append(seg_map)
            rgb = rgbs[rgb_idx]
        return np.stack(maps, axis=0)
    
    def _organize_seg_maps(
            self, 
            seg_maps: np.ndarray
        ) -> np.ndarray:
        ''' Organize segmentation maps in the form network will expect them.

            In particular, there will always be five maps: the first two for
            the lower garment (depending on the lower label), the second two
            for the upper garment (depending on the upper label), and the
            final for the whole clothed body.
        '''
        feature_maps = np.zeros((3, seg_maps.shape[1], seg_maps.shape[2]))
        feature_maps[-1] = seg_maps[0]
        feature_maps[0] = seg_maps[1]
        feature_maps[1] = seg_maps[2]
        return feature_maps
