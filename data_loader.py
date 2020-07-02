import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import pdb
import numpy as np
from PIL import Image
import random
import collections


import scipy
import scipy.ndimage
import scipy.sparse

import tarfile
from io import BytesIO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import rcParams

rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe'
#rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'



actions = [
        "None",
        "Drink",
        "Clapping",
        "Reading",
        "Phone call",
        "Interacting phone",
        "Bend",
        "Squad",
        "Wave",
        "Sitting",
        "Pointing to sth",
        "Lift/hold box",
        "Open drawer",
        "Pull/Push sth",
        "Eat from a plate",
        "Yarning /Stretch",
        "Kick"]


class SeqVolumeDataset(data.Dataset):
    def __init__(self, tar_fn, list_fn, seq_len, w = 61, h = 61, d = 85):
        self.tar_fn = tar_fn
        self.list_fn = list_fn
        self.seq_len = seq_len
        self.samples = self.load_list(list_fn)
        self.w = w
        self.h = h
        self.d = d
        self.rotation = True
        self.tar_fid = tarfile.open(self.tar_fn)
        self.name2member = {}
        self.duplicated_dict = {}
        for idx, member in enumerate(self.tar_fid.getmembers()):
            if idx % 3000 == 0:
                print(idx)
            if member.name.endswith("npz"):
                array_file = BytesIO()

                array_file.write(self.tar_fid.extractfile(member).read())
                array_file.seek(0)
                d = scipy.sparse.load_npz(array_file)
                if member.name.startswith('./'):
                    member.name = member.name.replace('./', '')
                self.name2member[member.name] = d


                path, fn = os.path.split(member.name)
                if fn not in self.duplicated_dict.keys():
                    self.duplicated_dict[fn] = []

                self.duplicated_dict[fn].append(member.name)
                self.duplicated_dict[fn].append(d)





    def load_list(self, list_fn):
        samples = []
        with open(list_fn, 'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                samples.append( (parts[0], int(parts[1]), int(parts[2])))
        return samples
                
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.samples[index]
        lbl = sample[2]
        vs = []
        if self.rotation:
            alpha = 2 * np.pi * np.random.rand()

        for i in range(self.seq_len):
            ffn = os.path.join(sample[0], '{:06d}'.format(sample[1] + i) + '.npz').replace('\\', '/')
            d = self.name2member[ffn]
            d = d.toarray()
            data = np.reshape(d, (self.w, self.h, self.d)).astype('float32')
            #data[data > 0 ] = 1.0
            #data *= 20
	
            #if self.rotation:
            #    data = scipy.ndimage.rotate(data, alpha, (0,1), reshape = False, order = 0, mode = 'nearest')
            
            p = np.random.uniform(0.7,1)
            r_idx = np.random.randint(6)
            mask = np.random.binomial(1, p, data.shape)
            #data = mask * data
            #if r_idx > 0:
            #    data[:,:,0:-r_idx] = data[:,:,r_idx:]
            #    data[:,:,-r_idx+1:] =  0
            data = data.astype('float32')
            vs.append(torch.from_numpy(data))

        return torch.stack(vs, dim = 0), torch.LongTensor([lbl])

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, lbls = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    return images, torch.stack(lbls, 0).squeeze(1)

def get_loader(tar_fn, list_fn, seq_len, num_workers, batch_size, w = 61, h = 61, d = 85, shuffle = True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    video_ds = SeqVolumeDataset(tar_fn = tar_fn,
                       list_fn = list_fn, 
                       seq_len = seq_len,
                       w = w, 
                       h = h, 
                       d = d)
    
    data_loader = torch.utils.data.DataLoader(dataset=video_ds,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

def update_plot(i, c, scat, ax_title):
    scat.set_array(c[i].ravel())
    ax_title.set_text('n_frame: {}'.format(i))
    return scat

if __name__ == '__main__':
    video_ds = SeqVolumeDataset(tar_fn = 'data/test3_cross_env/data_shell_f9_val.tar',
                       list_fn = 'data/test3_cross_env/val.lst',
                       seq_len = 10)
    #idx = random.randint(0, video_ds.__len__())
    for idx in range(1810, video_ds.__len__()):
    #for idx in range(5):
        np_video_ds, label = video_ds.__getitem__(idx)
        np_video_ds = np_video_ds.numpy()
        label = label.item()

        fig = plt.figure()
        fig.suptitle('Lable: {}'.format(actions[label]))

        frame_shape = np_video_ds[0].shape
        x = np.arange(frame_shape[0])[:, None, None]
        y = np.arange(frame_shape[1])[None, :, None]
        z = np.arange(frame_shape[2])[None, None, :]
        x, y, z = np.broadcast_arrays(x, y, z)

        c = np.array(np.tile(z, (np_video_ds.shape[0],1,1,1)), dtype='float32')
        c[np_video_ds == 0] = None

        ax = fig.gca(projection='3d')
        scat = ax.scatter(x.ravel(),
                   y.ravel(),
                   z.ravel(),
                   c=c[0].ravel(),
                    s=8,
                          marker='.',
                          depthshade=True)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax_title = ax.set_title('n_frame:')

        ani = animation.FuncAnimation(fig, update_plot, frames=range(np_video_ds.shape[0]),
                                fargs=(c, scat, ax_title))

        ani.save('test3_cross_env/{:06d}.gif'.format(video_ds.samples[idx][1]), writer='imagemagick', fps=5)



    #plt.show()


    pass
