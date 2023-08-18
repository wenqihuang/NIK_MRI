

import matplotlib
import numpy as np
import torch
from medutils.visualization import center_crop
from utils.mri import coilcombine, ifft2c_mri

def k2img(k, csm=None, im_size=None, norm_factor=1):
    """
    Convert k-space to image space
    :param k: k-space data on a Cartesian grid
    :param csm: coil sensitivity maps
    :return: image
    """

    coil_img = ifft2c_mri(k)
    if im_size is not None:
        coil_img = center_crop(coil_img, im_size)
        if csm is not None:
            csm = center_crop(csm, im_size)

    k_mag = k[:,4,:,:].abs().unsqueeze(1).detach().cpu().numpy()        # nt, nx, ny   
    # combined_img_motion = coil_img_motion.abs()
    if csm is not None:
        if len(csm.shape) == len(coil_img.shape):
            im_shape = csm.shape[2:]        # (nx, ny)
        else:
            im_shape = csm.shape[1:]        # (nx, ny)
        combined_img = coilcombine(coil_img, im_shape, coil_dim=1, csm=csm)
    else:
        combined_img = coilcombine(coil_img, coil_dim=1, mode='rss')
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    k_mag = np.log(np.abs(k_mag) + 1e-4)
    
    k_min = np.min(k_mag)
    k_max = np.max(k_mag)
    max_int = 255

    # combined_mag_nocenter = combined_mag
    # combined_mag_nocenter[:,:,combined_img.shape[-2]//2-10:combined_img.shape[-2]//2+10,combined_img.shape[-1]//2-10:combined_img.shape[-1]//2+10] = 0
    combined_mag_max = combined_mag.max() / norm_factor

    k_mag = (k_mag - k_min)*(max_int)/(k_max - k_min)
    k_mag = np.minimum(max_int, np.maximum(0.0, k_mag))
    k_mag = k_mag.astype(np.uint8)
    combined_mag = (combined_mag / combined_mag_max * 255)#.astype(np.uint8)
    combined_phase = angle2color(combined_phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
    combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
    combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)

    combined_img = combined_img.detach().cpu().numpy()
    vis_dic = {
        'k_mag': k_mag, 
        'combined_mag': combined_mag, 
        'combined_phase': combined_phase, 
        'combined_img': combined_img
    }
    return vis_dic

def angle2color(value_arr, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a value to a color using a colormap
    :param value: the value to convert
    :param cmap: the colormap to use
    :return: the color
    """
    if vmin is None:
        vmin = value_arr.min()
    if vmax is None:
        vmax = value_arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    try:
        value_arr = value_arr.squeeze(0)
    except:
        value_arr = value_arr.squeeze()
    if len(value_arr.shape) == 3:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            color_arr[i] = mapper.to_rgba(value_arr[i], bytes=True)
        color_arr = color_arr.transpose(0, 3, 1, 2)
    elif len(value_arr.shape) == 2:
        color_arr = mapper.to_rgba(value_arr, bytes=True)
    return color_arr
