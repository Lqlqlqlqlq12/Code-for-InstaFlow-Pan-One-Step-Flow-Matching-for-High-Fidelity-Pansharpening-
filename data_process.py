import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_h5_data(filepath, normalize=True, max_value = 2047.0, is_my = True):
    """读取 H5 文件并转换为归一化后的 numpy 数组。"""
    with h5py.File(filepath, 'r') as f:
        hrms = np.array(f['/gt'])
        lrms = np.array(f['/lms'])
        pan = np.array(f['/pan'])
        if is_my:
            pan = np.expand_dims(pan, axis=1)
        ms = np.array(f['/ms'])

    if normalize:
        hrms = hrms / max_value
        lrms = lrms / max_value
        pan = pan / max_value
        ms = ms / max_value

    print("hrms",hrms.shape)
    print("lms",lrms.shape)
    print("pan",pan.shape)
    print("ms",ms.shape)

    return np.clip(pan,0.0,1.0), np.clip(lrms,0.0,1.0), np.clip(hrms,0.0,1.0), np.clip(ms,0.0,1.0)

def load_test_h5_data(filepath, normalize=True, max_value = 2047.0, flag=True):
    """读取 H5 文件并转换为归一化后的 numpy 数组。"""
    with h5py.File(filepath, 'r') as f:
        hrms = np.array(f['/gt'])
        lrms = np.array(f['/lms'])
        pan = np.array(f['/pan'])
        if flag:
            pan = np.expand_dims(pan, axis=1)
        ms = np.array(f['/ms'])

    if normalize:
        hrms = hrms / max_value
        lrms = lrms / max_value
        pan = pan / max_value
        ms = ms / max_value

    print("hrms",hrms.shape)
    print("lms",lrms.shape)
    print("pan",pan.shape)
    print("ms",ms.shape)

    return np.clip(pan,0.0,1.0), np.clip(lrms,0.0,1.0), np.clip(hrms,0.0,1.0), np.clip(ms,0.0,1.0)

def load_h5_full_data(filepath, normalize=True, max_value = 2047.0, flag = True):
    """读取 H5 文件并转换为归一化后的 numpy 数组。"""
    with h5py.File(filepath, 'r') as f:
        lrms = np.array(f['/lms'])
        pan = np.array(f['/pan'])
        if flag:
            pan = np.expand_dims(pan, axis=1)
        ms = np.array(f['/ms'])

    if normalize:
        lrms = lrms / max_value
        pan = pan / max_value
        ms = ms / max_value

    print("lms",lrms.shape)
    print("pan",pan.shape)
    print("ms",ms.shape)

    return np.clip(pan,0.0,1.0), np.clip(lrms,0.0,1.0), np.clip(ms,0.0,1.0)




def prepare_full_dataloader(pan, lrms, ms, batch_size=8, shuffle=True, flag = 5):
    """将 numpy 数据转换为 Tensor 并构造 DataLoader。"""
    torch_pan  = torch.FloatTensor(pan)
    torch_lrms = torch.FloatTensor(lrms)
    torch_ms   = torch.FloatTensor(ms)

    if flag:
        torch_pan  = torch.FloatTensor(pan[0:flag,:,:,:])
        torch_lrms = torch.FloatTensor(lrms[0:flag,:,:,:])
        torch_ms   = torch.FloatTensor(ms[0:flag,:,:,:])

    dataset = TensorDataset(torch_pan, torch_lrms, torch_ms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def prepare_dataloader(pan, lrms, hrms, ms, batch_size=8, shuffle=True):
    """将 numpy 数据转换为 Tensor 并构造 DataLoader。"""
    torch_pan  = torch.FloatTensor(pan)
    torch_lrms = torch.FloatTensor(lrms)
    torch_hrms = torch.FloatTensor(hrms)
    torch_ms   = torch.FloatTensor(ms)

    # torch_pan  = torch.FloatTensor(pan[1:21,:,:,:])
    # torch_lrms = torch.FloatTensor(lrms[1:21,:,:,:])
    # torch_hrms = torch.FloatTensor(hrms[1:21,:,:,:])
    # torch_ms   = torch.FloatTensor(ms[1:21,:,:,:])

    dataset = TensorDataset(torch_pan, torch_lrms, torch_hrms, torch_ms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def build_train_loaders(train_path, max_value=[1023.0,2047.0,2047.0] ,batch_size=8, shuffle = True, is_my=True):
    """加载训练与验证集，并返回对应的 DataLoader。"""
    # 加载训练数据
    train_loader_list = []
    for index in range(len(train_path)):
        pan, lrms, hrms, ms = load_h5_data(train_path[index],max_value=max_value[index],is_my=is_my)
        train_loader = prepare_dataloader(pan, lrms, hrms, ms, batch_size=batch_size, shuffle=shuffle)
        print("path",train_path[index],"len",len(train_loader),"max_value",max_value[index])
        train_loader_list.append(train_loader)
    # # 加载验证数据
    # val_pan, val_lrms, val_hrms, val_ms = load_h5_data(val_path)
    # val_loader = prepare_dataloader(val_pan, val_lrms, val_hrms, val_ms, batch_size=batch_size, shuffle=False)
    return train_loader_list


def prepare_test_dataloader(pan, lrms, hrms, ms, batch_size=8, shuffle=True, flag = 5):
    """将 numpy 数据转换为 Tensor 并构造 DataLoader。"""
    torch_pan = torch.FloatTensor(pan)
    torch_lrms = torch.FloatTensor(lrms)
    torch_hrms = torch.FloatTensor(hrms)
    torch_ms = torch.FloatTensor(ms)

    if flag:
        torch_pan  = torch.FloatTensor(pan[0:flag,:,:,:])
        torch_lrms = torch.FloatTensor(lrms[0:flag,:,:,:])
        torch_hrms = torch.FloatTensor(hrms[0:flag,:,:,:])
        torch_ms   = torch.FloatTensor(ms[0:flag,:,:,:])

    # torch_pan  = torch.FloatTensor(pan)
    # torch_lrms = torch.FloatTensor(lrms)
    # torch_hrms = torch.FloatTensor(hrms)
    # torch_ms   = torch.FloatTensor(ms)

    dataset = TensorDataset(torch_pan, torch_lrms, torch_hrms, torch_ms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def build_test_loaders(train_path, max_value=[1023.0,2047.0,2047.0] ,batch_size=1, shuffle = False,  is_my=True, flag = 5):
    """加载训练与验证集，并返回对应的 DataLoader。"""
    # 加载训练数据
    train_loader_list = []
    for index in range(len(train_path)):
        pan, lrms, hrms, ms = load_test_h5_data(train_path[index],max_value=max_value[index], flag = is_my)
        train_loader = prepare_test_dataloader(pan, lrms, hrms, ms, batch_size=batch_size, shuffle=shuffle, flag=flag)
        print("path",train_path[index],"len",len(train_loader),"max_value",max_value[index])
        train_loader_list.append(train_loader)
    # # 加载验证数据
    # val_pan, val_lrms, val_hrms, val_ms = load_h5_data(val_path)
    # val_loader = prepare_dataloader(val_pan, val_lrms, val_hrms, val_ms, batch_size=batch_size, shuffle=False)
    return train_loader_list

def build_test_full_loaders(train_path, max_value=[1023.0,2047.0,2047.0] ,batch_size=1, shuffle = False, is_my=True, flag = False):
    """加载训练与验证集，并返回对应的 DataLoader。"""
    # 加载训练数据
    train_loader_list = []
    for index in range(len(train_path)):
        pan, lrms, ms = load_h5_full_data(train_path[index],max_value=max_value[index], flag = is_my)
        train_loader = prepare_full_dataloader(pan, lrms, ms, batch_size=batch_size, shuffle=shuffle, flag = flag)
        print("path",train_path[index],"len",len(train_loader),"max_value",max_value[index])
        train_loader_list.append(train_loader)
    # # 加载验证数据
    # val_pan, val_lrms, val_hrms, val_ms = load_h5_data(val_path)
    # val_loader = prepare_dataloader(val_pan, val_lrms, val_hrms, val_ms, batch_size=batch_size, shuffle=False)
    return train_loader_list


from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path, max_value):
        self.h5_file = h5py.File(h5_path, 'r')
        self.pan = np.array(self.h5_file['pan'], dtype=np.float32) / max_value
        self.pan = np.expand_dims(self.pan, axis=1)
        self.lrms = np.array(self.h5_file['lms'], dtype=np.float32) / max_value
        self.hrms = np.array(self.h5_file['gt'], dtype=np.float32) / max_value
        self.ms = np.array(self.h5_file['ms'], dtype=np.float32) / max_value
        self.length = self.pan.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.pan[idx]),
            torch.tensor(self.lrms[idx]),
            torch.tensor(self.hrms[idx]),
            torch.tensor(self.ms[idx])
        )

def create_for_three(path,max_value,batch_size):
    dataset1 = H5Dataset(path[0],max_value[0])
    dataset2 = H5Dataset(path[1],max_value[1])
    dataset3 = H5Dataset(path[2],max_value[2])

    # 合并 Dataset
    concat_dataset = ConcatDataset([dataset1, dataset2, dataset3])

    # 计算各自的长度
    len1, len2, len3 = len(dataset1), len(dataset2), len(dataset3)
    total_len = len1 + len2 + len3

    # 计算采样比例，按原始比例混合
    # 权重 = 1 / 数据集长度 → 总归一化
    weight1 = (1.0 / len1) * (len1 / total_len)
    weight2 = (1.0 / len2) * (len2 / total_len)
    weight3 = (1.0 / len3) * (len3 / total_len)

    # 为 concat_dataset 中每一条数据赋予采样权重
    weights = np.concatenate([
        np.full(len1, weight1),
        np.full(len2, weight2),
        np.full(len3, weight3)
    ])

    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=total_len, replacement=True)

    # DataLoader
    train_loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    return train_loader