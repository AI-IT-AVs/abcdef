import os.path as osp
import random

import numpy as np
import torch.utils.data

from posdiff.utils.common import load_pickle
from posdiff.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
)
from posdiff.utils.registration import get_correspondences

import h5py
import glob
import os
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
import pykitti

seq_id_set = ['boreas-2021-01-26-11-22', 'boreas-2021-04-29-15-55' , 'boreas-2021-06-17-17-52', 'boreas-2021-09-14-20-00', 'boreas-2021-05-13-16-11']
root_dir = '.../Boreas'

def jitter_pointcloud(pointcloud, sigma=0.0, clip=0.0):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def RT_Transform(R, T):
    T_matrix = np.eye(4)
    T_matrix[0:3,0:3] = R
    T_matrix[0:3,3] = T
    return T_matrix

def nearest_neighbor(dst, reserve):
    dst = dst.T
    num = np.max([dst.shape[0], dst.shape[1]])
    num = int(num * reserve)
    src = dst[-1, :].reshape(1, -1)
    neigh = NearestNeighbors(n_neighbors=num)
    neigh.fit(dst)
    indices = neigh.kneighbors(src, return_distance=False)
    indices = indices.ravel()
    return dst[indices, :].T

def transform_frame2_to_frame1(T_1, T_2): 
    T_tum2a = T_1
    T_tum2b = T_2
    T_ba = np.linalg.inv(T_tum2b) @ T_tum2a
    return T_ba

def load_data(partition='train', args=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../dataset')

    if args.dataset == 'kitti':
        all_idx = []
        rotations = []
        translations = []
        DATA_DIR = '/data/dataset/kitti/dataset/poses/'
        listTrain = ['09']
        listTest = ['10']
        root = DATA_DIR
        if partition == 'train':
            for seq in listTrain:
                T_seq = np.loadtxt(root+seq+'.txt')
                T_seq_temp = np.array([T_i.reshape(3,4) for T_i in T_seq])
                
                T_seq_indx = np.arange(T_seq_temp.shape[0]-1)
                T_seq_indx_next = T_seq_indx+1
                T_seq_name = np.ones(T_seq_indx.shape[0])*int(seq)
                idx_train = np.vstack((T_seq_name,T_seq_indx,T_seq_indx_next)).T
                idx_train = idx_train.astype('int32')
                
                frames = np.arange(T_seq_temp.shape[0])
                sequence = '{0:02d}'.format(int(seq))
                basedir = '/data/dataset/kitti/dataset'
                dataset0 = pykitti.odometry(basedir, sequence, frames=frames)
                
                velo2cam = dataset0.calib.T_cam0_velo   
                
                T_s_ba = []
                for iii in T_seq_indx:
                    T_s_a = T_seq_temp[iii]
                    T_s_b = T_seq_temp[iii+1]
                    r_a = T_s_a[0:3,0:3]
                    t_a = T_s_a[0:3,3]
                    r_b = T_s_b[0:3,0:3]
                    t_b = T_s_b[0:3,3]

                    T_a2utm = np.eye(4)
                    T_a2utm[0:3,0:3] = r_a
                    T_a2utm[0:3,3] = t_a
                    T_avelo2tum = np.matmul(T_a2utm, velo2cam)
                    T_b2utm = np.eye(4)
                    T_b2utm[0:3,0:3] = r_b
                    T_b2utm[0:3,3] = t_b
                    T_bvelo2tum = np.matmul(T_b2utm, velo2cam)
                    T_ba = np.matmul(np.linalg.inv(T_avelo2tum),T_bvelo2tum)
                    T_s_ba.append(T_ba)
                T_s_ba = np.array(T_s_ba)
                rotations_train = T_s_ba[:,0:3,0:3].astype('float32')
                translations_train = T_s_ba[:,0:3,3].astype('float32')
                all_idx.append(idx_train)
                rotations.append(rotations_train)
                translations.append(translations_train)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations
        else:
            for seq in listTest:
                T_seq = np.loadtxt(root+seq+'.txt')
                T_seq_temp = np.array([T_i.reshape(3,4) for T_i in T_seq])
                
                T_seq_indx = np.arange(T_seq_temp.shape[0]-1)
                T_seq_indx_next = T_seq_indx+1
                T_seq_name = np.ones(T_seq_indx.shape[0])*int(seq)
                idx_odo = np.vstack((T_seq_name,T_seq_indx,T_seq_indx_next)).T
                idx_odo = idx_odo.astype('int32')
                
                frames = np.arange(T_seq_temp.shape[0])
                sequence = '{0:02d}'.format(int(seq))
                basedir = '/data/dataset/kitti/dataset'
                dataset0 = pykitti.odometry(basedir, sequence, frames=frames)
                
                velo2cam = dataset0.calib.T_cam0_velo   
                
                T_s_ba = []
                for iii in T_seq_indx:
                    T_s_a = T_seq_temp[iii]
                    T_s_b = T_seq_temp[iii+1]
                    r_a = T_s_a[0:3,0:3]
                    t_a = T_s_a[0:3,3]
                    r_b = T_s_b[0:3,0:3]
                    t_b = T_s_b[0:3,3]

                    T_a2utm = np.eye(4)
                    T_a2utm[0:3,0:3] = r_a
                    T_a2utm[0:3,3] = t_a
                    T_avelo2tum = np.matmul(T_a2utm, velo2cam)
                    T_b2utm = np.eye(4)
                    T_b2utm[0:3,0:3] = r_b
                    T_b2utm[0:3,3] = t_b
                    T_bvelo2tum = np.matmul(T_b2utm, velo2cam)
                    T_ba = np.matmul(np.linalg.inv(T_avelo2tum),T_bvelo2tum)
                    T_s_ba.append(T_ba)
                T_s_ba = np.array(T_s_ba)
                
                rotations_odo = T_s_ba[:,0:3,0:3].astype('float32')
                translations_odo = T_s_ba[:,0:3,3].astype('float32')
                all_idx.append(idx_odo)
                rotations.append(rotations_odo)
                translations.append(translations_odo)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations
    elif args.dataset == 'boreas':
        listTrain_choice = args.listTrain_num
        listTest_choice = args.listTest_num

        all_idx = []
        rotations = []
        translations = []
        root = '/data/Boreas/sampled_lidar_pose'
        listTrain = [listTrain_choice]
        listTest =  [listTest_choice]
        if partition == 'train':
            for list_idx in listTrain:
                T_seq = np.load(os.path.join(root, seq_id_set[list_idx]+'_frame_pose.npy'), allow_pickle=True)
                T_seq_indx = np.arange(T_seq.shape[0]-1)
                T_seq_indx_next = T_seq_indx+1
                T_seq_name = np.ones(T_seq_indx.shape[0])*int(list_idx)
                idx_train = np.vstack((T_seq_name, T_seq_indx, T_seq_indx_next)).T
                idx_train = idx_train.astype('int32')

                T_s_ba = []
                for iii in T_seq_indx:
                    T_a = T_seq[iii]
                    T_b = T_seq[iii+1]
                    T_ba = transform_frame2_to_frame1(T_1=T_b, T_2=T_a)
                    T_s_ba.append(T_ba)
                T_s_ba = np.array(T_s_ba)
                rotations_train = T_s_ba[:,0:3,0:3].astype('float32')
                translations_train = T_s_ba[:,0:3,3].astype('float32')
                all_idx.append(idx_train)
                rotations.append(rotations_train)
                translations.append(translations_train)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations
        else:
            for list_idx in listTest:
                T_seq = np.load(os.path.join(root, seq_id_set[list_idx]+'_frame_pose.npy'), allow_pickle=True)
                T_seq_indx = np.arange(T_seq.shape[0]-1)
                T_seq_indx_next = T_seq_indx+1
                T_seq_name = np.ones(T_seq_indx.shape[0])*int(list_idx)
                idx_test = np.vstack((T_seq_name, T_seq_indx, T_seq_indx_next)).T
                idx_test = idx_test.astype('int32')

                T_s_ba = []
                for iii in T_seq_indx:
                    T_a = T_seq[iii]
                    T_b = T_seq[iii+1]
                    T_ba = transform_frame2_to_frame1(T_1=T_b, T_2=T_a)
                    T_s_ba.append(T_ba)
                T_s_ba = np.array(T_s_ba)
                rotations_test = T_s_ba[:,0:3,0:3].astype('float32')
                translations_test = T_s_ba[:,0:3,3].astype('float32')
                all_idx.append(idx_test)
                rotations.append(rotations_test)
                translations.append(translations_test)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations


def getPointCloud(seqN, binNum, binNumNext, R_ab=None, translation_ab=None, num_points=None):
    path = '/data/dataset/kitti/dataset/sampled/' + str(seqN).zfill(2) + '/'+str(binNum).zfill(6) + '.npy'
    pointcloud = np.load(path)[:, 0:3]
    points = pointcloud.shape[0]
    supply_idx = points // 6
    if points < num_points:
        point_supply = np.tile(pointcloud[supply_idx, :], (num_points - points, 1))
        pointcloud1 = np.concatenate((pointcloud, point_supply), axis=0)
    else:
        pointcloud1 = pointcloud[:num_points]
    return pointcloud1

##################################################################################################
'''
********* Boreas dataset processing *********
'''
##################################################################################################
def getPointCloud_boreas(seq_id_idx, binNum, num_points=None):
    seq_id_idx_file = glob.glob(os.path.join(root_dir, 'sampled_lidar', seq_id_set[seq_id_idx], '*.npy'))
    seq_id_idx_file=sorted(seq_id_idx_file)
    path = seq_id_idx_file[binNum]
    pointcloud = np.load(path, allow_pickle=True)[:, 0:3]
    points = pointcloud.shape[0]
    supply_idx = points // 6
    if points < num_points:
        point_supply = np.tile(pointcloud[supply_idx, :], (num_points - points, 1))
        pointcloud1 = np.concatenate((pointcloud, point_supply), axis=0)
    else:
        pointcloud1 = pointcloud[:num_points]
    return pointcloud1

class OdometryBoreasPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
    ):
        super(OdometryBoreasPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')
        
        self.reserve = cfg.reserve
        self.num_points = cfg.num_points
        self.partition = self.subset
        self.gaussian_noise = cfg.gaussian_noise
        self.partial = cfg.partial
        print('Load Boreas Dataset')
        self.all_idx, self.rotations, self.translations = load_data(self.subset, cfg)
        self.batch_size = cfg.train.batch_size
        self.test_batch_size = cfg.test.batch_size

    def __getitem__(self, index):
        data_dict = {}
        pointcloud_1_lidar = getPointCloud_boreas(self.all_idx[index, 0], self.all_idx[index, 1], num_points=int(self.num_points / self.reserve))
        pointcloud_2_lidar = getPointCloud_boreas(self.all_idx[index, 0], self.all_idx[index, 2], num_points=int(self.num_points / self.reserve))      

        pointcloud_2 = pointcloud_1_lidar
        pointcloud_1 = pointcloud_2_lidar  

        if self.partition != 'train':
            np.random.seed(index)

        R_ab = self.rotations[index,0:3,0:3]
        R_ba = R_ab.T
        translation_ab = self.translations[index,...]
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = np.random.permutation(pointcloud_1).T
        pointcloud2 = np.random.permutation(pointcloud_2).T
        r = Rotation.from_matrix(R_ab)
        euler_ab = r.as_euler(seq='zyx', degrees=True)
        euler_ab = np.radians(euler_ab)
        euler_ba = -euler_ab[::-1]

        if self.partial:
            pointcloud1 = nearest_neighbor(pointcloud1, self.reserve)
        pointcloud1 = pointcloud1[:, :self.num_points]
        pointcloud1 = np.random.permutation(pointcloud1.T).T

        if self.partial:
            pointcloud2 = nearest_neighbor(pointcloud2, self.reserve)
        pointcloud2 = pointcloud2[:, :self.num_points]
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        data_dict['seq_id'] = self.all_idx[index, 0]
        data_dict['ref_frame'] = self.all_idx[index, 1]
        data_dict['src_frame'] = self.all_idx[index, 1]

        ref_points = pointcloud2.T
        src_points = pointcloud1.T
        transform = RT_Transform(R_ab, translation_ab)


        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)


        return data_dict

    def __len__(self):
        return self.all_idx.shape[0]









