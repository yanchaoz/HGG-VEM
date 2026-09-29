from einops import rearrange,repeat
import torch
import torch.nn.functional as F
from skimage.feature import hog
from matplotlib import pyplot as plt
import random
import os
from einops import rearrange,repeat

def visual_2d(mae, img, save_dir, iters):
    with torch.no_grad():
        img,mask,recons = mae(img, vis=True)
        for i in range(5):
            x, y, z, _, _ = img.shape
            x_c, y_c, z_c = random.randint(0,x-1), random.randint(0,y-1), random.randint(0,z-1)
            plt.figure(figsize=(10,10))
            plt.subplot(1,3,1)
            plt.imshow(img[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.title('raw')

            plt.subplot(1,3,2)
            plt.imshow(mask[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.title('masked')

            plt.subplot(1,3,3)
            plt.imshow(recons[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.title('reconstructed')

            plt.savefig(os.path.join(save_dir, 'recons_{}_{}.png'.format(iters, i)), dpi=400, bbox_inches='tight')

    torch.cuda.empty_cache()
