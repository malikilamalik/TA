from __future__ import print_function, absolute_import

import random
from unicodedata import category
import torch.utils.data as data
from pose.utils.osutils import *
from pose.utils.transforms import *
from pose.utils.transforms import fliplr
from scipy.io import loadmat
import json
import csv
import argparse


class AP10_Animal_All(data.Dataset):
    def __init__(self, is_train=True, is_aug=False, **kwargs):
        print()
        print("==> AP10_Animal_All")
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.animal = ['antelope','argali', 'bison', 'horse', 'tiger', 'wolf', 'zebra'] if kwargs['animal'] == 'all' else [kwargs['animal']] # train on single or all animal categories
        self.train_on_all_cat = kwargs['train_on_all_cat']  # train on single or mul, decide mean file to load
        self.is_aug = is_aug

        # create train/val split
        self.train_img_set = []
        self.valid_img_set = []
        self.train_pts_set = []
        self.valid_pts_set = []
        self.load_animal()
        self.mean, self.std = self._compute_mean()

    def load_animal(self):
        # generate train/val data
        for animal in sorted(self.animal):
            img_list = []  # img_list contains all image paths
            img_list_train = []  # img_list_train contains train image paths
            img_list_valid = []  # img_list_valid contains valid image paths
            anno_list = []  # anno_list contains all anno lists
            anno_list_train = []  # anno_list contains train anno lists
            anno_list_valid = []  # anno_list contains valid anno lists
            landmark_path = os.path.join(self.img_folder, 'ap-10k/annotations')
            frame_num = 0
            test_path = ["ap10k-train-split1.json","ap10k-test-split1.json","ap10k-val-split1.json"]
            animal_category = 0
            if animal == 'antelope':
                animal_category = 1
            elif animal == 'argali_sheep':
                animal_category = 2
            elif animal == 'horse':
                animal_category = 21
            elif animal == 'cheetah':
                animal_category = 25
            elif animal == 'deer':
                animal_category = 17
            elif animal == 'buffalo':
                animal_category = 4
            elif animal == 'rabbit':
                animal_category = 38
            elif animal == 'squirrel':
                animal_category = 47
            elif animal == 'gorilla':
                animal_category = 36
            elif animal == 'monkey':
                animal_category = 13
            frame_num = 0   
            # for data_path in train_path:
            #     anno_path = os.path.join(landmark_path, data_path)
            #     # Opening JSON file
            #     f = open(anno_path)
                
            #     # returns JSON object as 
            #     # a dictionary
            #     data = json.load(f)
                
            #     # Iterating through the json
            #     # list
            #     for i,n in enumerate(data['annotations']):
            #         if n['category_id'] == animal_category:
            #             k = 0
            #             coord = []
            #             vis = []
            #             k_coord = []
            #             for i,l in enumerate(n['keypoints']):
            #                 if k != 2 :
            #                     k_coord.append(float(l))
            #                     k+=1
            #                 elif k == 2:
            #                     coord.append(k_coord)   
            #                     if l != 0 :
            #                         vis.append([1])
            #                     else:
            #                         vis.append([l])
            #                     k_coord = []
            #                     k=0
            #                 if i == 50:
            #                     coord.append([float(0),float(0)])
            #                     vis.append([0])
            #             landmark = np.hstack((coord, vis))
            #             landmark_18 = landmark[:18, :]
            #             landmark_18 = landmark_18[np.array([1, 2, 3, 8, 11, 14, 17, 5, 7, 10, 13, 16, 17, 17, 6, 9, 12, 15]) - 1]
            #             anno_list.append(landmark_18)
            #             anno_list_train.append(landmark_18)
            #             result_1 = str(n['image_id']).zfill(12)
            #             img_path_1 = "data/{}.jpg".format(result_1)
            #             img_list.append([img_path_1,0,0])
            #             img_list_train.append([img_path_1,0,0])
            #             self.train_img_set.append([img_path_1,0,0])
            #             self.train_pts_set.append(landmark_18)
            #             frame_num += 1
            #     # Closing file
            #     f.close() 
            for data_path in test_path:
                anno_path = os.path.join(landmark_path, data_path)
                # Opening JSON file
                f = open(anno_path)
                
                # returns JSON object as 
                # a dictionary
                data = json.load(f)
                
                # Iterating through the json
                # list
                for i,n in enumerate(data['annotations']):
                    if n['category_id'] == animal_category:
                        k = 0
                        coord = []
                        vis = []
                        k_coord = []
                        for i,l in enumerate(n['keypoints']):
                            if k != 2 :
                                k_coord.append(float(l))
                                k+=1
                            elif k == 2:
                                coord.append(k_coord)   
                                if l != 0 :
                                    vis.append([1])
                                else:
                                    vis.append([l])
                                k_coord = []
                                k=0
                            if i == 50:
                                coord.append([float(0),float(0)])
                                vis.append([0])
                        landmark = np.hstack((coord, vis))
                        landmark_18 = landmark[:18, :]
                        landmark_18 = landmark_18[np.array([1, 2, 3, 8, 11, 14, 17, 5, 7, 10, 13, 16, 17, 17, 6, 9, 12, 15]) - 1]
                        anno_list.append(landmark_18)
                        anno_list_valid.append(landmark_18)
                        result_1 = str(n['image_id']).zfill(12)
                        img_path_1 = "data/{}.jpg".format(result_1)
                        img_list.append([img_path_1,0,0])
                        img_list_valid.append([img_path_1,0,0])
                        frame_num += 1
                # Closing file
                f.close()

            # train_idxs = np.load('./data/real_animal/' + animal + '/train_idxs_by_video.npy')
            data_valid_id_filename = './data/ap10k/animal_{}_id.csv'.format(animal)
            with open(data_valid_id_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for idx in csv_reader:
                    valid_idx = int(idx[0])
                    self.valid_img_set.append(img_list[valid_idx])
                    self.valid_pts_set.append(anno_list[valid_idx])
            # for video in range_file['ranges']:
            #     # range_file['ranges'] is a numpy array [Nx3]: shot_id, start_frame, end_frame
            #     shot_id = video[0]
            #     landmark_path_video = os.path.join(landmark_path, str(shot_id) + '.mat')

            #     if not os.path.isfile(landmark_path_video):
            #         continue
            #     landmark_file = loadmat(landmark_path_video)

            #     for frame in range(video[1], video[2] + 1):  # ??? video[2]+1
            #         frame_id = frame - video[1]
            #         img_name = animal + '/' + '0' * (8 - len(str(frame))) + str(frame) + '.jpg'
            #         img_list.append([img_name, shot_id, frame_id])

            #         coord = landmark_file['landmarks'][frame_id][0][0][0][0]
            #         vis = landmark_file['landmarks'][frame_id][0][0][0][1]
            #         landmark = np.hstack((coord, vis))
            #         landmark_18 = landmark[:18, :]
            #         if animal == 'horse':
            #             anno_list.append(landmark_18)
            #         elif animal == 'tiger':
            #             landmark_18 = landmark_18[
            #                 np.array([1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12]) - 1]
            #             anno_list.append(landmark_18)
            #         frame_num += 1
            # # data_filename_in_csv_format = './animal_{}.csv'.format(animal)
            # # data_filename_in_csv_format_train = './animal_{}_train.csv'.format(animal)
            # # data_filename_in_csv_format_test = './animal_{}_test.csv'.format(animal)
            # # anno_filename_in_csv_format_train = './animal_{}_train_anno.csv'.format(animal)
            # # anno_filename_in_csv_format_test = './animal_{}_test_anno.csv'.format(animal)
            # # with open(str(data_filename_in_csv_format), 'w', newline='') as myfile:
            # #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # #     for i in img_list:
            # #         wr.writerow(i)
            # for idx in range(train_idxs.shape[0]):
            #     train_idx = train_idxs[idx]
            #     img_list_train.append(img_list[train_idx])
            #     anno_list_train.append(anno_list[train_idx])
            #     self.train_img_set.append(img_list[train_idx])
            #     self.train_pts_set.append(anno_list[train_idx])
            # for idx in range(valid_idxs.shape[0]):
            #     valid_idx = valid_idxs[idx]
            #     img_list_valid.append(img_list[valid_idx])
            #     anno_list_valid.append(anno_list[valid_idx])
            #     self.valid_img_set.append(img_list[valid_idx])
            #     self.valid_pts_set.append(anno_list[valid_idx])
            # # with open(str(data_filename_in_csv_format_train), 'w', newline='') as myfile:
            # #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # #     for i in img_list_train:
            # #         wr.writerow(i)
            # # with open(str(anno_filename_in_csv_format_train), 'w', newline='') as myfile:
            # #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # #     for i in anno_list_train:
            # #         wr.writerow(i)
            # # with open(str(data_filename_in_csv_format_test), 'w', newline='') as myfile:
            # #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # #     for i in img_list_valid:
            # #         wr.writerow(i)
            # # with open(str(anno_filename_in_csv_format_test), 'w', newline='') as myfile:
            # #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # #     for i in anno_list_valid:
            # #         wr.writerow(i)
            print('Animal:{}, number of frames:{}, train: {}, valid: {}'.format(animal, frame_num,
                                                                         len(img_list_train), len(self.valid_img_set)))
        print('Total number of frames:{}, train: {}, valid {}'.format(len(img_list), len(self.train_img_set),
                                                                      len(self.valid_img_set)))

    def _compute_mean(self):
        animal = 'all' if self.train_on_all_cat else self.animal[0]  # which mean file to load
        meanstd_file = './data/synthetic_animal/' + animal + '_combineds5r5_texture' + '/mean.pth.tar'

        if isfile(meanstd_file):
            print('load from mean file:', meanstd_file)
            meanstd = torch.load(meanstd_file)
        else:
            print("generate mean file")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train_list:
                a = self.img_list[index][0]
                img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', a)
                img = load_image_ori(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        print('  Real animal  mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('  Real animal  std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        img_list = self.train_img_set if self.is_train else self.valid_img_set
        anno_list = self.train_pts_set if self.is_train else self.valid_pts_set
        try:
            a = img_list[index][0]
        except IndexError:
            print(index)

        img_path = os.path.join(self.img_folder, 'ap-10k', a)
        img = load_image_ori(img_path)  # CxHxW
        pts = anno_list[index].astype(np.float32)
        x_vis = pts[:, 0][pts[:, 0] > 0]
        y_vis = pts[:, 1][pts[:, 1] > 0]

        try:
            # generate bounding box using keypoints
            height, width = img.size()[1], img.size()[2]
            y_min = float(max(np.min(y_vis) - 15, 0.0))
            y_max = float(min(np.max(y_vis) + 15, height))
            x_min = float(max(np.min(x_vis) - 15, 0.0))
            x_max = float(min(np.max(x_vis) + 15, width))
        except ValueError:
            print(img_path, index)
        # Generate center and scale for image cropping,
        # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        nparts = pts.shape[0]
        pts = torch.Tensor(pts)
        r = 0
        if self.is_aug and self.is_train:
            # print('augmentation')
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr_ori(pts, width=img.size(2), dataset='real_animal')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        tpts_inpres = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                tpts_inpres[i, 0:2] = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'target_weight': target_weight, 'pts_256': tpts_inpres}
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_img_set)
        else:
            return len(self.valid_img_set)


def ap10_animal_all(**kwargs):
    return AP10_Animal_All(**kwargs)


ap10_animal_all.njoints = 18  # ugly but works

