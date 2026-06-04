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
