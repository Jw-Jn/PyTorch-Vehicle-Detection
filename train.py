import numpy as np
import torch.nn
import torch.optim as optim
import cityscape_dataset as csd
from torch.utils.data import Dataset
from torch.autograd import Variable
from bbox_helper import generate_prior_bboxes, match_priors
from data_loader import get_list
from ssd_net import SSD
from bbox_loss import MultiboxLoss
import matplotlib.pyplot as plt
import os
import random

use_gpu = True
img_dir = '../../../../../Courses_data/datasets/full_dataset/train_extra/'
label_dir = '../../../../../Courses_data/datasets/full_dataset_labels/train_extra/'
save_dir = '../../../../../Courses_data/'
learning_rate = 0.001
max_epoches = 150

train_list = get_list(img_dir, label_dir)
random.shuffle(train_list)
train_list = train_list[:1000]
train_dataset = csd.CityScapeDataset(train_list, train=True, show=False)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                num_workers=0)
print('train items:', len(train_dataset))
print('max_epoches:', max_epoches)
file_name = 'csd_mobilenet_cropped_random_'+str(len(train_list))+'_'+str(max_epoches)+'epc'

net = SSD(3)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
criterion = MultiboxLoss([0.1,0.1,0.2,0.2])

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.cuda()
    criterion.cuda()

train_losses = []
valid_losses = []
conf_losses = []
loc_huber_losses = []
itr = 0
for epoch_idx in range(0, max_epoches):
    for train_batch_idx, (loc_targets, conf_targets, imgs) in enumerate(train_data_loader):
       
        itr += 1
        net.train()

        imgs = imgs.permute(0, 3, 1, 2).contiguous() # [batch_size, W, H, CH] -> [batch_size, CH, W, H]
        if use_gpu:
            imgs = imgs.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        imgs = Variable(imgs)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        conf_preds, loc_preds = net.forward(imgs)
        conf_loss, loc_huber_loss, loss = criterion.forward(conf_preds, loc_preds, conf_targets, loc_targets)
        
        loss.backward()
        optimizer.step()
        
        if train_batch_idx % 100 == 0:
            print('Epoch: %d Itr: %d Conf_Loss: %f Loc_Loss: %f Loss: %f' 
            % (epoch_idx, itr, conf_loss.item(), loc_huber_loss.item(), loss.item()))
            train_losses.append((itr, loss.item()))
            conf_losses.append((itr, conf_loss.item()))
            loc_huber_losses.append((itr, loc_huber_loss.item()))
        
    if epoch_idx==100:
        net_state = net.state_dict()
        torch.save(net_state, os.path.join(save_dir, file_name+'-'+str(epoch_idx)+'.pth'))
        print(file_name+'-'+str(epoch_idx)+'.pth saved')

train_losses = np.asarray(train_losses)
conf_losses = np.asarray(conf_losses)
loc_huber_losses = np.asarray(loc_huber_losses)
plt.plot(train_losses[:, 0],
         train_losses[:, 1])
plt.plot(conf_losses[:, 0],
         conf_losses[:, 1])
plt.plot(loc_huber_losses[:, 0],
         loc_huber_losses[:, 1])
# plt.show()
plt.savefig(file_name+'.jpg')

net_state = net.state_dict()
torch.save(net_state, os.path.join(save_dir, file_name+'.pth'))