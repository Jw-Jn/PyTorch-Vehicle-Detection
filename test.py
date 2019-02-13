import os
import torch.nn
import numpy as np
import cityscape_dataset as csd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
from torch.autograd import Variable
from bbox_helper import loc2bbox, nms_bbox, bbox2loc
from data_loader import get_list
from ssd_net import SSD
from PIL import Image

use_gpu = False
img_dir = '../../../../../Courses_data/datasets/full_dataset/train_extra/'
label_dir = '../../../../../Courses_data/datasets/full_dataset_labels/train_extra/'
save_dir = '../../../../../Courses_data/'
train_list_len = 1000
max_epoches = 150
file_name = 'csd_mobilenet_cropped_random_'+str(train_list_len)+'_'+str(max_epoches)+'epc'

test_list = get_list(img_dir, label_dir)
test_list = test_list[0:10]
test_dataset = csd.CityScapeDataset(test_list, train=False, show=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0)
print('test items:', len(test_dataset))

net = SSD(3)

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.cuda()

net_state = torch.load(os.path.join(save_dir, file_name+'.pth'))
net.load_state_dict(net_state)
net.eval()

for test_batch_idx, (loc_targets, conf_targets, imgs) in enumerate(test_data_loader):
    imgs = imgs.permute(0, 3, 1, 2).contiguous()
    if use_gpu:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    conf, loc = net.forward(imgs)
    conf = conf[0,...]
    loc = loc[0,...].cpu()
    
    prior =  test_dataset.get_prior_bbox()
    prior = torch.unsqueeze(prior, 0)
    real_bounding_box = loc2bbox(loc, prior)
    real_bounding_box = torch.squeeze(real_bounding_box)

    class_list, sel_box = nms_bbox(real_bounding_box, conf, overlap_threshold=0.5, prob_threshold=0.4)
    # print(len(sel_box))

    img = imgs[0].permute(1, 2, 0).contiguous()
    img = Image.fromarray(np.uint8(img*128 + 127))
    sel_box = np.array(sel_box)

    loc_targets = torch.squeeze(loc2bbox(loc_targets, prior)).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # predict, human--blue, vehicle--red
    for idx in range(len(sel_box)):
        cx, cy, w, h = sel_box[idx]*300
        if class_list[idx] == 1:
            rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='r', facecolor='none')
        if class_list[idx] == 2:
            rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # ground truth--green
    mask = np.where(conf_targets[0]>0)
    for idx in mask[0]:
        cx, cy, w, h = loc_targets[idx]*300
        rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.show()