import os
import os.path
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# manual_seed=123
# torch.manual_seed(manual_seed)
# np.random.seed(manual_seed)
# torch.cuda.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)
# random.seed(manual_seed)
# os.environ['PYTHONHASHSEED'] = str(manual_seed) 


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None, data_split=0, shot=10, seed=123, \
    sub_list=None, sub_val_list=None):
    assert split in ['train', 'val', 'val_supp']
    manual_seed = seed
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)    
    data_split=data_split

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    sub_class_file_list = {}
    for sub_c in sub_val_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])              
        item = (image_name, label_name)

        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        if len(label_class) == 0:
            continue

        flag = 'keep'  
        for c in label_class:
            if c in sub_val_list:
                sub_class_file_list[c].append((image_name, label_name))
                flag = 'drop'
                break  

        if flag == 'keep' and split != 'val_supp':
            item = (image_name, label_name)
            image_label_list.append(item)
    print("Checking Pretrain image&label pair {} list {} done!".format(split, data_split))
    print("All {} pairs in base classes.".format(len(image_label_list)))

    supp_image_label_list = []
    if (split == 'train' or split == 'val_supp'):
        shot = shot
        for c in sub_val_list:
            sub_class_file = sub_class_file_list[c]
            num_file = len(sub_class_file)
            output_data_list = []
            select_list = []
            num_trial = 0
            while(len(select_list) < shot):
                num_trial += 1
                if num_trial >= num_file:
                    print('class {} skip with {} shots'.format(c, len(select_list)))
                    raw_select_list = select_list.copy()
                    for re in range(shot - len(select_list)):
                        rand_select_idx = raw_select_list[random.randint(0,len(raw_select_list)-1)]
                        select_list.append(rand_select_idx)
                        supp_image_label_list.append(sub_class_file[rand_select_idx])
                        output_data_list.append(sub_class_file[rand_select_idx][1].split('/')[-1])                            
                    break
                rand_idx = random.randint(0,num_file-1)
                if rand_idx in select_list:
                    continue
                else:              
                    label = cv2.imread(sub_class_file[rand_idx][1], cv2.IMREAD_GRAYSCALE)
                    label_class = np.unique(label).tolist()
                    label_class.remove(c) 
                    if 0 in label_class:
                        label_class.remove(0)
                    if 255 in label_class:
                        label_class.remove(255)       

                    skip_flag = 0
                    for new_c in label_class:
                        if new_c in sub_val_list:
                            skip_flag = 1
                            break
                    if skip_flag:
                        continue
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    if target_pix[0].shape[0] >= 16 * 32 * 32 and target_pix[1].shape[0] >= 16 * 32 * 32:                       
                        select_list.append(rand_idx)
                        supp_image_label_list.append(sub_class_file[rand_idx])
                        output_data_list.append(sub_class_file[rand_idx][1].split('/')[-1])
                    else:
                        continue
    else:
        ### for 'val' mode that evaluates all images
        for c in sub_val_list:
            sub_class_file = sub_class_file_list[c]
            for idx in range(len(sub_class_file)):
                supp_image_label_list.append(sub_class_file[idx])           

    image_label_list = supp_image_label_list + image_label_list
    print("Checking image&label pair {} list done!".format(split))
    print("All {} pairs in novel classes.".format(len(supp_image_label_list)))
    print("All {} pairs in base + novel classes.".format(len(image_label_list)))
    return image_label_list, supp_image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, \
        transform=None, data_split=0, shot=10, seed=123, \
        use_coco=False, val_shot=10):

        if use_coco:
            print('INFO: using COCO')
            self.class_list = list(range(1, 81))
            if data_split == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
            elif data_split == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))  
            elif data_split == 11:
                self.sub_list = list(range(41, 81)) 
                self.sub_val_list = list(range(1, 41))                  
            elif data_split == 10:
                self.sub_list = list(range(1, 41)) 
                self.sub_val_list = list(range(41, 81))                 
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)
             
        else:
            # use PASCAL VOC | 0-20 + 255
            print('INFO: using PASCAL VOC')
            if data_split == 3:  
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = [16,17,18,19,20]
            elif data_split == 2:
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = [11,12,13,14,15]
            elif data_split == 1:
                self.sub_list = [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [6,7,8,9,10]
            elif data_split == 0:
                self.sub_list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [1,2,3,4,5]                
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)

        print('sub_list: ',self.sub_list)
        print('sub_val_list: ',self.sub_val_list)
        print('Base_num: {} (including class 0), Novel_num: {}'.format(self.base_class_num, self.novel_class_num))
        save_dir = './saved_npy/'
        if use_coco:
            path_np_data_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list_coco.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list_coco.npy'
        else:
            path_np_data_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list.npy'
        path_np_data_list = os.path.join(save_dir, path_np_data_list)
        path_np_supp_list = os.path.join(save_dir, path_np_supp_list)
        if not os.path.exists(path_np_data_list):
            print('[{}] Creating new lists and will save to **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list, self.supp_image_label_list = make_dataset(split, data_root, data_list, data_split=data_split, shot=shot, seed=seed, \
                                            sub_list=self.sub_list, sub_val_list=self.sub_val_list)  
            np_data_list = np.array(self.data_list)
            np_supp_list = np.array(self.supp_image_label_list)
            np.save(path_np_data_list, np_data_list)
            np.save(path_np_supp_list, np_supp_list)
        else:
            print('[{}] Loading saved lists from **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list = list(np.load(path_np_data_list))
            self.supp_image_label_list = list(np.load(path_np_supp_list))

        print('Processing data list {} with {} shots.'.format(data_split, shot))
        self.data_split=data_split
        self.transform = transform
        self.split = split
        self.use_coco = use_coco

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]                   
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
            
        raw_label = label.copy()
        for c in label_class:
            x,y = np.where(raw_label == c)
  
            if c in self.sub_list:
                label[x[:], y[:]] = (self.sub_list.index(c) + 1)    # ignore the background in sublist, + 1
            elif c in self.sub_val_list:
                label[x[:], y[:]] = (self.sub_val_list.index(c) + self.base_class_num)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        raw_size = torch.Tensor(label.shape[:])
        raw_label = label.copy()
        raw_label_mask = np.zeros((1024, 1024))
        raw_label_mask[:raw_label.shape[0], :raw_label.shape[1]] = raw_label.copy()
        raw_label_mask = torch.Tensor(raw_label_mask)
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, raw_size, raw_label_mask
