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
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

import csv


rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe'
# rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'


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
    def __init__(self, tar_fn, list_out_fn, seq_len=10, w=61, h=61, d=85):
        self.tar_fn = tar_fn
        self.list_out_fn = list_out_fn
        self.seq_len = seq_len
        #self.samples = self.load_list(list_fn)
        self.w = w
        self.h = h
        self.d = d
        self.rotation = True
        self.tar_fid = tarfile.open(self.tar_fn)
        self.name2member = {}
        self.duplicated_dict = {}
        self.sequence_list = [member for member in self.tar_fid.getmembers() if member.name.endswith('npz')]
        self.sequence_list.sort(key=lambda member: member.name)

    def load_list(self, list_fn):
        samples = []
        with open(list_fn, 'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                samples.append((parts[0], int(parts[1]), int(parts[2])))
        return samples

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        data = []

        for offset in range(self.seq_len):
            array_file = BytesIO()
            array_file.write(self.tar_fid.extractfile(self.sequence_list[index + offset]).read())
            array_file.seek(0)
            d = scipy.sparse.load_npz(array_file)
            d = d.toarray()
            d = np.reshape(d, (self.w, self.h, self.d)).astype('float32')
            data.append(d)

            array_file.close()


        return np.array(data)

    def __len__(self):
        return self.sequence_list.__len__() - self.seq_len

    def get_name(self, index):
        return self.sequence_list[index].name

    def write_label(self, index, label):
        # 메모리에 저장해놨다가 한번에 flush하면 좋을 듯
        list_out_fid = open(self.list_out_fn, 'w+')
        list_out_fid.write('{0} {1}'.format(self.sequence_list[index].name, label))
        list_out_fid.close()

    def write_titles(self):
        with open(self.list_out_fn, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')

            for idx, member in enumerate(self.sequence_list):
                writer.writerow([member.name])


if __name__ == '__main__':
    video_ds = SeqVolumeDataset(tar_fn='data/test3_cross_env/data_shell_f9_val_refined.tar',
                                list_out_fn='data/test3_cross_env/val_refined.csv',
                                seq_len=10)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax_title = ax.set_title('')

    class Index(object):
        def __init__(self):
            self.idx = 0
            self.current_data = video_ds.__getitem__(self.idx)

        def next(self, event):
            self.idx += 1
            if self.idx > video_ds.__len__():
                self.idx = video_ds.__len__()

            self.current_data = video_ds.__getitem__(self.idx)
            self.draw()

        def prev(self, event):
            self.idx -= 1
            if self.idx < 0:
                self.idx = 0

            self.current_data = video_ds.__getitem__(self.idx)
            self.draw()

        def submit(self, text):
            self.idx = int(text)
            if self.idx < 0:
                self.idx = 0
            elif self.idx > video_ds.__len__():
                self.idx = video_ds.__len__()

            self.current_data = video_ds.__getitem__(self.idx)
            self.draw()

        def draw(self):
            ax_title.set_text('Original label: {}'.format(video_ds.get_name(self.idx)))

            frame_shape = self.current_data[0].shape
            x = np.arange(frame_shape[0])[:, None, None]
            y = np.arange(frame_shape[1])[None, :, None]
            z = np.arange(frame_shape[2])[None, None, :]
            x, y, z = np.broadcast_arrays(x, y, z)

            c = np.array(np.tile(z, (self.current_data.shape[0], 1, 1, 1)), dtype='float32')
            c[self.current_data == 0] = None

            scat = ax.scatter(x.ravel(),
                              y.ravel(),
                              z.ravel(),
                              c=c[0].ravel(),
                              s=8,
                              marker='.',
                              depthshade=True)


            def update_plot(i, c, scat, ax_title):
                scat.set_array(c[i].ravel())
                ax_title.set_text('Original label: {}'.format(video_ds.get_name(self.idx + i)))
                return scat

            self.ani = animation.FuncAnimation(fig, update_plot, frames=range(self.current_data.shape[0]), fargs=(c, scat, ax_title), interval=200)

            #ani.save('test3_cross_env/{:06d}.gif'.format(video_ds.samples[idx][1]), writer='imagemagick', fps=5)

            plt.show()


    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)

    axbox = plt.axes([0.1, 0.05, 0.1, 0.075])
    text_box = TextBox(axbox, 'Index')
    text_box.on_submit(callback.submit)

    callback.draw()

    pass
