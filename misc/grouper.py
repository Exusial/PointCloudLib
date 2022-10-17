import jittor as jt
from jittor import nn
from misc.ops import BallQueryGrouper

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx
        self.query = BallQueryGrouper(self.radius, self.nsample)

    def execute(self, query_xyz, support_xyz, features=None):
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        if features is not None:
            features = features.permute(0,2,1)
        new_features, idx = self.query(query_xyz, support_xyz, features)

        if self.return_only_idx:
            return idx
        grouped_xyz = support_xyz.reindex([support_xyz.shape[0], 3, query_xyz.shape[1], self.nsample],
        ['i0', '@e0(i0, i2, i3)', 'i1'],extras=[idx])
        # xyz_trans = support_xyz.transpose(1, 2)
        # grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius       
        return grouped_xyz, new_features.permute(0,3,1,2)

class GroupAll_Trans(nn.Module):
    def __init__(self, use_xyz):
        super().__init__()
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        if self.use_xyz:
            new_feature = jt.concat([pointset.permute(0,2,1), feature], dim=-1)
        else:
            new_feature = feature
        return new_xyz.permute(0,2,1).unsqueeze(2), new_feature.unsqueeze(2)