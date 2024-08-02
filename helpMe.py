import os
import torch
import torchvision.utils as vu
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
import torchvision
import torchvision.transforms as T


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def  to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def denorm(img_tensors, stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64, fig_size=[8,8]):
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(vu.make_grid(denorm(images.detach()[:nmax]), nrow=fig_size[1]).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, var in dl:
        show_images(images, nmax)
        break

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def to_gaus(imgs):
    smoothed_imgs = []
    higher_freq = []

    for img_tensor in imgs:

        img = T.ToPILImage()(img_tensor)
         
        S_img = img.filter(ImageFilter.GaussianBlur(radius=4))  # Adjust the radius as needed
              
        # H_img = T.ToPILImage()(H_img)
        S_img = T.ToTensor()(S_img)
        H_img = img_tensor - S_img
        higher_freq.append(H_img)
        smoothed_imgs.append(S_img)

    smoothed_imgs = torch.stack(smoothed_imgs)
    higher_freq = torch.stack(higher_freq)
    torchvision.utils.save_image(smoothed_imgs.detach(), f"smoooooo.png", normalize=True,nrow=8)
    return smoothed_imgs,higher_freq





def save_generated_images(genH_realH, recon=None, epoch=0, i=0, path='', res='0',a=''):
    # print("save path",path )
    os.makedirs(f"{path}/Generated/LAPGan", exist_ok=True)
    torchvision.utils.save_image(genH_realH.detach().to('cpu'), f"{path}Generated/{a}{epoch}_{i}_generated_images_{res}.png", normalize=True, nrow=8)
    if recon is not None:
        torchvision.utils.save_image(recon.detach().to('cpu'), f"{path}Generated/{epoch}_{i}_generated_recon_{res}.png", normalize=True, nrow=8)

