import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction



class PointNetV2(nn.Module):
    def __init__(self):  # remove the num_class premeter
        super(PointNetV2, self).__init__()
        normal_channel = False
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel  # default: false
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        return x


if __name__ == '__main__':
    input = torch.rand([10, 1024, 3])
    net = PointNetV2()
    a = net(input)
    print(a.shape)
