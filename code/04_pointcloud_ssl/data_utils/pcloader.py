# import h5py
# import numpy as np
# import warnings
# import os
# import data_utils.augmentation as augmentation
# from torch.utils.data import Dataset
# import torch

# warnings.filterwarnings('ignore')


# def pc_normalize(pc):
#     # print('-----------------pc',pc.shape)
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     # print('-----------------pc_cen', pc.shape)
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
#     # print('-----------------pc_nor', pc.shape)
#     return pc


# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:, :3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point


# class ModelNetDataLoader(Dataset):
#     def __init__(self, root, npoint=1024, uniform=False, istrain=True):

#         self.npoints = npoint
#         self.root = root
#         self.len = len(os.listdir(self.root))

#         self.uniform = uniform
#         self.istrain = istrain

#     def __len__(self):
#         return self.len

#     def _get_item(self, index):
#         point_set = np.loadtxt(os.path.join(self.root, '%04d.txt' % index), delimiter=',').astype(np.float32)

#         if self.istrain:

#             point_view1 = augmentation.transform(point_set)
#             point_view2 = augmentation.transform(point_set)

#             point_view1 = farthest_point_sample(point_view1, self.npoints)
#             point_view2 = farthest_point_sample(point_view2, self.npoints)

#             point_view1[:, 0:3] = pc_normalize(point_view1[:, 0:3])
#             point_view2[:, 0:3] = pc_normalize(point_view2[:, 0:3])

#             # point_view1 = point_view1[:, 0:3]
#             # point_view2 = point_view2[:, 0:3]

#             # point_view1 = torch.tensor(point_view1, dtype=torch.float32)
#             # point_view2 = torch.tensor(point_view2, dtype=torch.float32)

#             return (point_view1, point_view2)

#         else:

#             point_set = farthest_point_sample(point_set, self.npoints)

#             point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

#             point_set = point_set[:, 0:3]

#             return point_set

#     def __getitem__(self, index):
#         return self._get_item(index)


# def visualizePointCloud(points, savePath):
#     """
#     Input:
#         points: pointcloud data, [N, D]
#     """
#     x = [k[0] for k in points]
#     y = [k[1] for k in points]
#     z = [k[2] for k in points]
#     fig = plt.figure(dpi=500)
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title('Points cloud')
#     ax.scatter(x, y, z, c='b', marker='.', s=10, linewidth=0, alpha=1, cmap='spectral')
#     plt.savefig(savePath)


# if __name__ == '__main__':

#     import torch
#     from matplotlib import pyplot as plt
#     import augmentation as augmentation

#     data = ModelNetDataLoader('./mito/', uniform=True, istrain=True)

#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)

#     for (point_view1, point_view2) in DataLoader:
#         point = point_view1.numpy()
#         for i in range(len(point)):
#             visualizePointCloud(point[i], 'test' + str(i) + '.png')
#         print(point.shape)

import h5py
import numpy as np
import warnings
import os
import data_utils.augmentation as augmentation
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    # print('-----------------pc',pc.shape)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # print('-----------------pc_cen', pc.shape)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    # print('-----------------pc_nor', pc.shape)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, uniform=False, istrain=True, holdout_last_n=16):

        self.npoints = npoint
        self.file = h5py.File(root, 'r')
        self.data = np.array(self.file['cloudpoints'])
        if holdout_last_n:
            self.data = self.data[:-holdout_last_n]
        if len(self.data) == 0:
            raise ValueError('No point clouds remain after holdout_last_n')

        self.uniform = uniform
        self.istrain = istrain

    def __len__(self):
        return self.data.shape[0]

    def _get_item(self, index):
        point_set = self.data[index]

        if self.istrain:

            point_view1 = augmentation.transform(point_set)
            point_view2 = augmentation.transform(point_set)

            point_view1 = farthest_point_sample(point_view1, self.npoints)
            point_view2 = farthest_point_sample(point_view2, self.npoints)

            point_view1[:, 0:3] = pc_normalize(point_view1[:, 0:3])
            point_view2[:, 0:3] = pc_normalize(point_view2[:, 0:3])

            point_view1 = point_view1[:, 0:3]
            point_view2 = point_view2[:, 0:3]

            point_view1 = torch.tensor(point_view1, dtype=torch.float32)
            point_view2 = torch.tensor(point_view2, dtype=torch.float32)

            return (point_view1, point_view2)
        else:

            point_set = farthest_point_sample(point_set, self.npoints)

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            # point_set = point_set[:, 0:3]
            point_set = torch.tensor(point_set, dtype=torch.float32)

            return point_set

    def __getitem__(self, index):
        return self._get_item(index)


class Provider(object):
    def __init__(self, cfg):
        self.data = ModelNetDataLoader(cfg.DATA.pointcloud_h5, npoint=1024, uniform=True, istrain=True,
                                      holdout_last_n=cfg.DATA.holdout_last_n)
        self.batch_size = cfg.TRAIN.batch_size
        self.num_workers = cfg.TRAIN.num_workers
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        self.data_iter = iter(
            DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                       shuffle=True, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch


def visualizePointCloud(points, savePath):
    """
    Input:
        points: pointcloud data, [N, D]
    """
    x = [k[0] for k in points]
    y = [k[1] for k in points]
    z = [k[2] for k in points]
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Points cloud')
    ax.scatter(x, y, z, c='b', marker='.', s=10, linewidth=0, alpha=1, cmap='spectral')
    plt.savefig(savePath)


if __name__ == '__main__':

    import torch
    from matplotlib import pyplot as plt
    import augmentation as augmentation

    data = ModelNetDataLoader('united.h5', uniform=True, istrain=True)

    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)

    for (point_view1, point_view2) in DataLoader:
        point = point_view1.numpy()
        for i in range(len(point)):
            visualizePointCloud(point[i], 'test' + str(i) + '.png')
        print(point.shape)
