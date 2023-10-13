from typing import Dict
import numpy as np
import os
import cv2


def load_poseshape(img_name):
    pass


def load_segmaps(
        img_name: str,
        mask_dir: str
    ) -> Dict[str, np.ndarray]:
    masks_dict = {}
    for part_label in ['lower-cloth', 'upper-cloth', 'whole-body']:
        masks_dict[part_label] = cv2.imread(os.path.join(
            mask_dir,
            f'{img_name.split(".")[0]}_{part_label}.png'
        ))
    return masks_dict
