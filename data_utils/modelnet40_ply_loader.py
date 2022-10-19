import os
import subprocess
from pathlib import Path
import glob
import h5py
import numpy as np
from tqdm import tqdm
import logging
import jittor as jt
from jittor.dataset.dataset import Dataset

BASE_DIR = Path(__file__).parent

def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path


def load_data(data_dir, partition, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    print(all_data.shape, all_label.shape)
    return all_data, all_label

class ModelNet40_h5(Dataset):
    dir_name = 'modelnet40_ply_hdf5_2048'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(self,
                 n_points=1024,
                 data_dir="./data/ModelNet40Ply2048",
                 train=True,
                 batch_size=1,
                 transform=None,
                 shuffle=False
                 ):
        super().__init__()
        if data_dir is not None:
            data_dir = os.path.join(
                os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        else:
            data_dir = os.path.join(BASE_DIR, "ModelNet40Ply2048")
        self.partition = 'train' if train else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.partition, self.url)
        self.n_points = n_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def collect_batch(self, batch):
        pts = np.stack([b[0] for b in batch], axis=0)
        cls = np.stack([b[2] for b in batch])
        return pts, cls

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.n_points]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = jt.concat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data['pos'], jt.array([]), data['y']

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1
    
    def normalize_pointclouds(self, pts):
        pts = pts - pts.mean(axis=0)
        scale = np.sqrt((pts ** 2).sum(axis=1).max())
        pts = pts / scale
        return pts

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

if __name__ == '__main__':
    modelnet40 = ModelNet40(n_points=2048, train=True, batch_size=32, shuffle=True)
    for pts, normals, cls in modelnet40:
        print (pts.size(), normals.size())
        break
