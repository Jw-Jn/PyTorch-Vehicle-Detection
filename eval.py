import os
import torch.nn
import numpy as np
import cityscape_dataset as csd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.autograd import Variable
from bbox_helper import loc2bbox, nms_bbox, generate_prior_bboxes
from ssd_net import SSD
from PIL import Image
import sys

def img2tensor(img_path):
    img = Image.open(img_path)
    img = img.resize((300, 300))

    mean = np.asarray((127, 127, 127))
    std = 128.0

    sample_img = (np.array(img, dtype=np.float) - mean) / std
    img_tensor = torch.from_numpy(sample_img).float()
    img_tensor = torch.unsqueeze(img_tensor, 0)

    return img_tensor

use_gpu = False
img_path = sys.argv[1]
file_name = 'csd_mobilenet_cropped_random_1000_150epc.pth'

net = SSD(3)

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.cuda()

net_state = torch.load(file_name)
net.load_state_dict(net_state)
net.eval()

img = img2tensor(img_path)
img = img.permute(0, 3, 1, 2).contiguous()
if use_gpu:
    img = img.cuda()
img = Variable(img)
conf, loc = net.forward(img)
conf = conf[0]
loc = loc[0]

prior = generate_prior_bboxes()
prior = torch.unsqueeze(prior, 0)
real_bounding_box = loc2bbox(loc, prior)
real_bounding_box = torch.squeeze(real_bounding_box)

class_list, sel_box = nms_bbox(real_bounding_box, conf, overlap_threshold=0.5, prob_threshold=0.4)
# print(len(sel_box))

img = img[0].permute(1, 2, 0).contiguous()
img = Image.fromarray(np.uint8(img*128 + 127))
sel_box = np.array(sel_box)

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

plt.show()
