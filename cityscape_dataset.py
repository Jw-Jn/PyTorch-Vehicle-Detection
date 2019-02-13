import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class CityScapeDataset(Dataset):
    img_size = 300

    def __init__(self, dataset_list, train, show):
        self.dataset_list = dataset_list

        # implement prior bounding box
        self.prior_bboxes = generate_prior_bboxes()

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0
        self.train = train
        self.show = show

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """
        
        # implement data loading
        # 1. Load image as well as the bounding box with its label
        item = self.dataset_list[idx]
        img_path = item['img']
        h = item['h']
        w = item['w']
        sample_labels = item['labels']
        sample_bboxes_corner = item['bboxes']
        img = Image.open(img_path)
        
        # data augment
        if self.train:
            img, sample_bboxes_corner = self.random_flip(img, sample_bboxes_corner)
            # img, boxes, labels = self.random_crop(img, boxes, labels)

        # crop
        img, sample_bboxes_corner, sample_labels = self.crop_img(img, sample_bboxes_corner, sample_labels, img_path, h)
        img.save('augsburg_cropped.png')
        
        img = img.resize((self.img_size, self.img_size))

        # Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        lt = sample_bboxes_corner[:,0,:]
        rb = sample_bboxes_corner[:,1,:]
        wh = rb-lt
        c = (lt + wh/2)
        sample_bboxes = np.stack((c[:,0]/h, c[:,1]/h, wh[:,0]/h, wh[:,1]/h), axis = 1) # crop
        # sample_bboxes = np.stack((c[:,0]/w, c[:,1]/h, wh[:,0]/w, wh[:,1]/h), axis = 1)  # no crop

        # Normalize the image with self.mean and self.std
        sample_img = (np.array(img, dtype=np.float) - self.mean) / self.std
        img_tensor = torch.from_numpy(sample_img).float()
        sample_bboxes = torch.from_numpy(np.asarray(sample_bboxes)).float()
        sample_labels = torch.from_numpy(np.asarray(sample_labels)).float()

        # matching prior, generate ground-truth labels and boxes
        bbox_tensor, bbox_label_tensor, bbox_offset_tensor = match_priors(self.prior_bboxes.cpu(), sample_bboxes, sample_labels, iou_threshold=0.5)

        if self.show:
            self.show_bbox(img, sample_bboxes.numpy(), self.prior_bboxes.cpu().numpy(), bbox_label_tensor.numpy())

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_offset_tensor, bbox_label_tensor, img_tensor

    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            xmin = img.width - boxes[:,1,0]
            xmax = img.width - boxes[:,0,0]
            boxes[:,0,0] = xmin
            boxes[:,1,0] = xmax
        return img, boxes

    def crop_img(self, img, sample_bboxes_corner, sample_labels, img_path, crop_h):
        lt = sample_bboxes_corner[:,0,:]
        rb = sample_bboxes_corner[:,1,:]
        wh = rb - lt

        step = [0, 682, 1024]
        crop = []
        crop_label = []
        for i in step:
            crop_lt = np.tile(np.array([i, 0]), (sample_bboxes_corner.shape[0], 1))
            crop_rb = np.tile(np.array([i+crop_h, crop_h]), (sample_bboxes_corner.shape[0], 1))
            inter_lt = np.maximum(lt, crop_lt)
            inter_rb = np.minimum(rb, crop_rb)
            mask = wh - (inter_rb - inter_lt)
            crop.append(sum((mask[:,0] - mask[:,1])==0)) # num of prior box for each crop_bbox
            crop_label.append(np.where((mask[:,0] - mask[:,1])!=0)) # label which is not in the crop_bbox

        # choose crop position
        arg = np.argmax(crop)
        img_crop = img.crop((step[arg], 0, step[arg]+crop_h, crop_h))
        delete_list = crop_label[arg][0]
        sample_labels_c = np.delete(sample_labels, delete_list)
        sample_bboxes_corner_c = np.delete(sample_bboxes_corner, delete_list, 0)
        offset = np.tile(np.array([step[arg], 0]), (sample_bboxes_corner_c.shape[0], 2, 1))
        sample_bboxes_corner_c -= offset

        if len(sample_labels_c) == 0:
            left = lt[0][0]
            w = wh[0][0]
            xmin = left - (crop_h - w)//2 # xmin of the crop
            img_crop = img.crop((xmin, 0, xmin+crop_h, crop_h))
            offset = np.tile(np.array([xmin, 0]), (1, 2, 1))
            sample_bboxes_corner_c = np.expand_dims(sample_bboxes_corner[0], axis=0)-offset
            sample_labels_c = np.expand_dims(sample_labels[0], axis=0)

        return img_crop, sample_bboxes_corner_c, sample_labels_c

    
    def show_bbox(self, img, gt_bbox, prior_bbox, bbox_label):
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for bbox in gt_bbox:
            cx, cy, w, h = bbox*self.img_size
            rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        mask = np.where(bbox_label>0)
        # print(sum(mask))
        for idx in mask[0]:
            cx, cy, w, h = prior_bbox[idx]*self.img_size
            rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        plt.show()