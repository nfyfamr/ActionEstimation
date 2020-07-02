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

import csv

from traits.api import HasTraits, Range, Instance, on_trait_change, Array, Tuple, Str, Int, Button
from traitsui.api import View, Item, HGroup, Handler
from traitsui.key_bindings import KeyBinding, KeyBindings
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.tools.animator import animate
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.modules.axes import Axes
from numpy import linspace, pi, cos, sin
from mayavi import mlab




def curve(n_mer, n_long):
    phi = linspace(0, 2*pi, 2000)
    return [ cos(phi*n_mer) * (1 + 0.5*cos(n_long*phi)),
            sin(phi*n_mer) * (1 + 0.5*cos(n_long*phi)),
            0.5*sin(n_long*phi),
            sin(phi*n_mer)]


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
    def __init__(self, tar_fn, list_fn, list_out_fn='data/test3_cross_env/val_refined.csv', seq_len=10, w=61, h=61, d=85):
        self.tar_fn = tar_fn
        self.list_fn = list_fn
        self.list_out_fn = list_out_fn
        self.seq_len = seq_len
        self.samples = self.load_list(list_fn)
        self.w = w
        self.h = h
        self.d = d
        self.rotation = True
        self.tar_fid = tarfile.open(self.tar_fn)
        self.name2member = {}
        self.duplicated_dict = {}
        self.sequence_list = [member for member in self.tar_fid.getmembers() if member.name.endswith('npz')]
        self.sequence_list.sort(key=lambda member: member.name)

        for index, member in enumerate(self.sequence_list):
            array_file = BytesIO()
            array_file.write(self.tar_fid.extractfile(self.sequence_list[index]).read())
            array_file.seek(0)
            d = scipy.sparse.load_npz(array_file)
            d = d.toarray()
            d = np.reshape(d, (self.w, self.h, self.d)).astype('float32')
            self.name2member[member.name] = d

    def load_list(self, list_fn):
        samples = {}
        if list_fn.endswith('.lst'):
            with open(list_fn, 'r') as fid:
                for aline in fid:
                    parts = aline.strip().split()
                    parts[0] = parts[0].split('/')
                    parts[0] = '/'.join(parts[0][:-1])
                    samples['{}/{:06d}.npz'.format(parts[0], int(parts[1]))] = parts[2]

        elif list_fn.endswith('.csv'):
            with open(list_fn, 'r') as fid:
                csv_reader = csv.reader(fid, delimiter=',')
                for row in csv_reader:
                    samples[row[0]] = [int(label) if label != '' else 0 for label in row[2:]]

        return samples

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        return torch.from_numpy(self.name2member[self.get_name(index)]), torch.LongTensor(self.samples[self.get_name(index)])

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
                if member.name in self.samples:
                    writer.writerow([member.name, ',', self.samples[member.name]])
                else:
                    writer.writerow([member.name])

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


class Visualization(HasTraits):
    seq_num = Int(0, desc='Sequence number', auto_set=False, enter_set=True)
    seq_name = Str('0')
    next_seq = Button('Next seq')
    prev_seq = Button('Prev seq')
    scene      = Instance(MlabSceneModel, ())

    def __init__(self, data_set):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.data_set = data_set

        x, y, z, s = self.render(0)
        self.plot = self.scene.mlab.points3d(x, y, z, s, colormap='hot', scale_factor=1, scale_mode='none')
        self.trait_set(seq_name=self.data_set.get_name(0))
        # self.anim(self))
        #mlab.axes(figure=self.scene.mayavi_scene)

    def key_down(self, vtk, event):
        vtk.GetKeyCode()

    def _next_seq_fired(self):
        seq_num = int( getattr(self, 'seq_num') )
        seq_num += 1

        if seq_num > self.data_set.__len__():
            seq_num = self.data_set.__len__()

        x, y, z, s = self.render(seq_num)
        self.scene.mlab.clf()
        self.plot = self.scene.mlab.points3d(x, y, z, s, colormap='hot', scale_factor=1, scale_mode='none')

        self.trait_set(seq_num=seq_num, seq_name=self.data_set.get_name(seq_num))
        #self.plot.mlab_source.trait_set(seq_num=seq_num)

    def _prev_seq_fired(self):
        seq_num = int( getattr(self, 'seq_num') )
        seq_num -= 1

        if seq_num < 0:
            seq_num = 0

        x, y, z, s = self.render(self.seq_num)
        self.scene.mlab.clf()
        self.plot = self.scene.mlab.points3d(x, y, z, s, colormap='hot', scale_factor=1, scale_mode='none')
        self.trait_set(seq_num=seq_num, seq_name=self.data_set.get_name(seq_num))

    @on_trait_change('seq_num')
    def update_seq_num(self):
        seq_num = int( getattr(self, 'seq_num') )

        x, y, z, s = self.render(seq_num)
        self.scene.mlab.clf()
        self.plot = self.scene.mlab.points3d(x, y, z, s, colormap='hot', scale_factor=1, scale_mode='none')
        self.trait_set(seq_name=self.data_set.get_name(seq_num))
        # self.anim(self)


    @animate(delay=100)
    def anim(self):
        for i in range(10):
            frame = self.seq[i]
            x, y, z = np.nonzero(frame)
            s = np.linspace(0, 1, num=x.shape[0])

            self.scene.mlab.clf()
            self.plot = self.scene.mlab.points3d(x, y, z, s, colormap='hot', scale_factor=1, scale_mode='none')
            # self.plot.mlab_source.trait_set(x=x, y=y, z=z, s=s)
            # self.plot.mlab_source.scalars = np.asarray(x * 0.1 * (i + 1), 'd')
            yield

    def render(self, index):
        self.seq, val = self.data_set[index]
        self.seq = self.seq.numpy()
        val = val.numpy()
        first_frame = self.seq
        x, y, z = np.nonzero(first_frame)
        s = np.linspace(0, 1, num=x.shape[0])
        return x, y, z, s

    def _LeftKeyPressed(self, event):
        self._prev_seq_fired(self)

    def _RightKeyPressed(self, event):
        self._next_seq_fired(self)

    key_bindings = KeyBindings(
        KeyBinding(binding1='Left',
                   description='prev seq',
                   method_name='_LeftKeyPressed'),
        KeyBinding(binding1='Right',
                   description='next seq',
                   method_name='_RightKeyPressed')
    )

    # class KeyHandler(Handler):
    #
    #     def save_file(self, info):
    #         info.object.status = "save file"
    #
    #     def run_script(self, info):
    #         info.object.status = "run script"
    #
    #     def edit_bindings(self, info):
    #         info.object.status = "edit bindings"
    #         key_bindings.edit_traits()

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=1000, width=1200, show_label=False),
                HGroup(Item('seq_num'),
                       Item('prev_seq'),
                       Item('next_seq')),
                HGroup(Item('seq_name', style='readonly')),
                key_bindings=key_bindings,
                resizable=True
                )



def update_plot(i, c, scat, ax_title):
    scat.set_array(c[i].ravel())
    ax_title.set_text('n_frame: {}'.format(i))
    return scat

if __name__ == '__main__':
    video_ds = SeqVolumeDataset(tar_fn='data/test3_cross_env/data_f9.tar',
                                list_fn='data/test3_cross_env/val_refined_001529.csv',
                                list_out_fn='data/test3_cross_env/val_refined.csv',
                                seq_len=10)

    # video_ds[0].shape => (61, 61, 85)
    # video_ds.write_titles()

    video_ds[0]
    visualization = Visualization(video_ds)
    visualization.configure_traits()





    # #idx = random.randint(0, video_ds.__len__())
    # for idx in range(1810, video_ds.__len__()):
    # #for idx in range(5):
    #     np_video_ds, label = video_ds.__getitem__(idx)
    #     np_video_ds = np_video_ds.numpy()
    #     label = label.item()
    #
    #     fig = plt.figure()
    #     fig.suptitle('Lable: {}'.format(actions[label]))
    #
    #     frame_shape = np_video_ds[0].shape
    #     x = np.arange(frame_shape[0])[:, None, None]
    #     y = np.arange(frame_shape[1])[None, :, None]
    #     z = np.arange(frame_shape[2])[None, None, :]
    #     x, y, z = np.broadcast_arrays(x, y, z)
    #
    #     c = np.array(np.tile(z, (np_video_ds.shape[0],1,1,1)), dtype='float32')
    #     c[np_video_ds == 0] = None
    #
    #     ax = fig.gca(projection='3d')
    #     scat = ax.scatter(x.ravel(),
    #                y.ravel(),
    #                z.ravel(),
    #                c=c[0].ravel(),
    #                 s=8,
    #                       marker='.',
    #                       depthshade=True)
    #
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    #     ax_title = ax.set_title('n_frame:')
    #
    #     ani = animation.FuncAnimation(fig, update_plot, frames=range(np_video_ds.shape[0]),
    #                             fargs=(c, scat, ax_title))
    #
    #     ani.save('test3_cross_env/{:06d}.gif'.format(video_ds.samples[idx][1]), writer='imagemagick', fps=5)
    #
    #
    #
    # #plt.show()
    #
    #
    # pass
