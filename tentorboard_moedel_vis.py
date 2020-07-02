import torch
from torch.utils.tensorboard import SummaryWriter
from data_loader_mayavi import SeqVolumeDataset
from model_advanced import ActionNet



# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('runs/tensorboard_1')


trainset = SeqVolumeDataset(tar_fn='data/test3_cross_env/data_f9.tar',
                            list_fn='data/test3_cross_env/val_refined_001529.csv',
                            seq_len=1, w = 61, h = 61, d= 85)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

net = ActionNet(512, 10)

writer.add_graph(net, images)
writer.close()