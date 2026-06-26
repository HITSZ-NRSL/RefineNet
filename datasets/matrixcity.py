import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

class MATRIXCITY(torch.utils.data.Dataset):
    """加载预处理NPY文件的数据集，支持训练增强 & 测试固定稀疏点采样"""
    def __init__(self, npy_dir="/home/qfy/temp/wyk/big_city_npy/big_city/aerial", 
                 split='train', fixed_num_points=None, seed=42, p_augment=0.7):
        self.samples = []
        self.split = split
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.fixed_num_points = fixed_num_points
        self.rng = np.random.RandomState(seed)
        self.p_augment = p_augment

        # 相机内参
        self.Kcam = torch.tensor([
            [618.0,   0.0, 256.0],
            [  0.0, 618.0, 128.0],
            [  0.0,   0.0,   1.0]
        ], dtype=torch.float32)

        npy_dir = os.path.join(npy_dir, split)
        # 遍历所有子文件夹收集样本路径
        for block_dir in sorted(os.listdir(npy_dir)):
            block_path = os.path.join(npy_dir, block_dir)
            if not os.path.isdir(block_path):
                continue
            for file in sorted(os.listdir(block_path)):
                if file.endswith("_rgb.npy"):
                    base_name = file[:-8]  # 去掉'_rgb.npy'
                    base_path = os.path.join(block_path, base_name)
                    if split != 'train':
                        # 测试集加载 sparse 文件检查是否有非零像素
                        sparse_check = np.load(f"{base_path}_sparse.npy")
                        if np.count_nonzero(sparse_check) == 0:
                            continue # 跳过没有点的样本
                    self.samples.append(base_path)

        print(f"Loaded {len(self.samples)} preprocessed samples from {npy_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_path = self.samples[idx]

        # ===== 1. 加载数据 =====
        rgb = np.load(f"{base_path}_rgb.npy")          # [H,W,3] in [0,1]
        depth = np.load(f"{base_path}_depth.npy")      # [H,W]
        sparse = np.load(f"{base_path}_sparse.npy")    # [H,W]

        # 转为 Tensor
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()   # [3,H,W]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()   # [1,H,W]
        sparse_tensor = torch.from_numpy(sparse).unsqueeze(0).float() # [1,H,W]

        Kcam = self.Kcam.clone()

        # ===== 2. 数据增强 (train only) =====
        if self.split == 'train' and random.random() < self.p_augment:
            aug_ops = []

            # 1. 随机水平翻转
            def flip_op(rgb, depth, sparse):
                rgb_f = TF.hflip(rgb)
                dep_f = TF.hflip(depth)
                sp_f = TF.hflip(sparse)
                Kcam[0,2] = rgb.shape[2] - 1 - Kcam[0,2]  # cx 更新
                return rgb_f, dep_f, sp_f
            aug_ops.append(lambda: flip_op(rgb_tensor, depth_tensor, sparse_tensor))

            # 2. 随机亮度/对比度
            def color_op(rgb, depth, sparse):
                rgb_aug = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
                rgb_aug = TF.adjust_contrast(rgb_aug, random.uniform(0.8, 1.2))
                return rgb_aug, depth, sparse
            aug_ops.append(lambda: color_op(rgb_tensor, depth_tensor, sparse_tensor))

            # 3. 稀疏点随机采样
            def resample_op(rgb, depth, sparse):
                num_points = random.choice([0, 5, 10, 50, 100, 500, 1000])
                nonzero_coords = torch.nonzero(sparse.squeeze())
                if num_points == 0:
                    new_sparse = torch.zeros_like(sparse)
                elif len(nonzero_coords) > num_points:
                    selected_indices = torch.randperm(len(nonzero_coords))[:num_points]
                    selected_coords = nonzero_coords[selected_indices]
                    new_sparse = torch.zeros_like(sparse)
                    for (h, w) in selected_coords:
                        new_sparse[0, h, w] = sparse[0, h, w]
                else:
                    new_sparse = sparse
                return rgb, depth, new_sparse
            aug_ops.append(lambda: resample_op(rgb_tensor, depth_tensor, sparse_tensor))

            # 4. 稀疏点 Dropout
            def dropout_op(rgb, depth, sparse):
                drop_mask = (torch.rand_like(sparse) < 0.1).float()
                return rgb, depth, sparse * (1 - drop_mask)
            aug_ops.append(lambda: dropout_op(rgb_tensor, depth_tensor, sparse_tensor))

            # 5. 局部遮挡
            def erase_op(rgb, depth, sparse):
                H, W = sparse.shape[1:]
                h0, w0 = random.randint(0, H // 2), random.randint(0, W // 2)
                hh, ww = random.randint(10, H // 4), random.randint(10, W // 4)
                sparse_clone = sparse.clone()
                sparse_clone[:, h0:h0+hh, w0:w0+ww] = 0.0
                return rgb, depth, sparse_clone
            aug_ops.append(lambda: erase_op(rgb_tensor, depth_tensor, sparse_tensor))

            # 随机选择一种增强操作
            rgb_tensor, depth_tensor, sparse_tensor = random.choice(aug_ops)()

        # ===== 3. 测试固定采样 =====
        elif self.fixed_num_points is not None:
            num_points = self.fixed_num_points
            nonzero_coords = torch.nonzero(sparse_tensor.squeeze())
            self.rng.seed(idx)
            if num_points == 0:
                sparse_tensor = torch.zeros_like(sparse_tensor)
            elif len(nonzero_coords) > num_points:
                selected_indices = self.rng.choice(len(nonzero_coords), num_points, replace=False)
                selected_coords = nonzero_coords[selected_indices]
                new_sparse = torch.zeros_like(sparse_tensor)
                for (h, w) in selected_coords:
                    new_sparse[0, h, w] = sparse_tensor[0, h, w]
                sparse_tensor = new_sparse

        # ===== 4. 归一化 RGB =====
        rgb_norm = TF.normalize(rgb_tensor, self.norm_mean, self.norm_std)

        return rgb_norm, sparse_tensor, Kcam, depth_tensor
