import numpy as np
import jittor as jt

class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=1,
                 append_xyz=False,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - jt.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim + 1]
                data['heights'] = height - jt.min(height)

            if self.centering:
                data['pos'] = data['pos'] - jt.mean(data['pos'], 0, keepdims=True)

            if self.normalize:
                m = jt.max(jt.sqrt(jt.sum(data['pos'] ** 2, -1, keepdims=True)), 0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - jt.mean(data, -1, keepdims=True)
            if self.normalize:
                m = jt.max(jt.sqrt(jt.sum(data ** 2, -1, keepdims=True)), 0, keepdims=True)[0]
                data = data / m
        return data

class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = jt.array(np.array(mirror))
        self.use_mirroring = jt.sum(jt.array(self.mirror)>0) != 0

    def __call__(self, data):
        scale = jt.rand(3 if self.anisotropic else 1) * (self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            mirror = (jt.rand(3) > self.mirror).float32() * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        data *= scale
        return data

TRANSFORM_MAP = {
    "PointCloudCenterAndNormalize": PointCloudCenterAndNormalize,
    "PointCloudScaling": PointCloudScaling
}

def get_transform(type, **kwargs):
    return TRANSFORM_MAP[type](**kwargs)