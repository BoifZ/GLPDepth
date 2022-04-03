import os
import cv2

from dataset.base_dataset import BaseDataset


class kitti(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/', 
                 is_train=True, dataset='kitti', crop_size=(352, 704), image_size=(896, 1184),
                 scale_size=None, mode='None'):
        super().__init__(crop_size)        

        self.image_size = image_size
        self.scale_size = scale_size
        
        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'kitti')

        # self.image_path_list = []
        # self.depth_path_list = []
        txt_path = os.path.join(filenames_path, 'eigen_benchmark')
        
        # if  mode is not None:
        txt_path += '/train_list.txt' if is_train else '/test_list.txt'
        filenames_list = self.readTXT(txt_path)
        self.filenames_list = []
        
        for f in filenames_list:
            if os.path.exists(self.data_path+f.split(' ')[0]):
                self.filenames_list.append(f)

        ## for custom_data 
        # from glob import glob
        # self.data_path = '/home/boif/Desktop/NeuS/data/mid_close/preprocessed'
        # pre = len(self.data_path)
        # filenames_list = sorted(glob(os.path.join(self.data_path, 'image/*.png')))
        # for f in filenames_list:
        #     f = f[pre:]
        #     self.filenames_list.append(f+' '+f.replace('image', 'depth'))
        
        # print(self.filenames_list)
        phase = 'train' if is_train else 'test'
        print("Dataset :", dataset)
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    # kb cropping
    def cropping(self, img, crop_size=(352, 1216)):
        h_im, w_im = img.shape[:2]

        margin_top = int(h_im - crop_size[0])
        margin_left = int((w_im - crop_size[1]) / 2)

        img = img[margin_top: margin_top + crop_size[0],
                  margin_left: margin_left + crop_size[1]]
        return img

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        # print(img_path)
        filename = img_path.split('/')[-4] + '_' + img_path.split('/')[-1]
        image = cv2.imread(img_path)  # [H x W x C] and C: BGR
        print(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        image = self.cropping(image, crop_size=self.image_size)
        depth = self.cropping(depth, crop_size=self.image_size)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 256.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': filename}


def gen_file_lists(images_path, depths_path):
    with open('syn_train_list.txt', 'w') as record:
        for f in os.listdir(images_path):
            depth_f = os.path.join(depths_path, f[:-4]+'_depth.png')
            if os.path.exists(depth_f):
                record.write(os.path.join(images_path, f), ' ', depth_f)


if __name__=='__main__':
    images_path = ''
    depths_path = ''
    gen_file_lists()