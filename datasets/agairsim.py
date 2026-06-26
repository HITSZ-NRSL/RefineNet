import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class AGAIRSIM(Dataset):
    """
    适配 MatrixCity 格式的 AirSim 数据集
    支持按场景划分训练/测试集，并自动适配相机内参缩放。
    """
    def __init__(self, root, split='train', img_size=(256, 384), 
                 test_scenes=['SimTown_2'],
                 fixed_num_points=None, seed=42, p_augment=0.7):
        self.root = root
        self.split = split
        self.img_size = img_size # (H, W)
        self.fixed_num_points = fixed_num_points
        self.p_augment = p_augment
        self.rng = np.random.RandomState(seed)
        
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        # 1. 收集所有路径并按场景过滤
        all_rgb_paths = sorted(glob.glob(os.path.join(root, "*", "*", "rgb", "*.png")))
        self.rgb_paths = []
        
        for path in all_rgb_paths:
            # 检查路径是否包含指定的测试场景名称
            is_test_scene = any(scene in path for scene in test_scenes)
            
            if split == 'train':
                if not is_test_scene:
                    self.rgb_paths.append(path)
            else: # val or test
                if is_test_scene:
                    self.rgb_paths.append(path)
            
        print(f"[{split.upper()}] Loaded {len(self.rgb_paths)} samples. Test scenes: {test_scenes}")

        # 2. 相机内参缩放逻辑
        orig_W, orig_H = 768, 512
        orig_fx = orig_fy = 384.64645201
        orig_cx, orig_cy = 384.0, 256.0
        
        scale_w = img_size[1] / orig_W
        scale_h = img_size[0] / orig_H
        
        self.base_K = torch.tensor([
            [orig_fx * scale_w,  0.0,               orig_cx * scale_w],
            [0.0,                orig_fy * scale_h, orig_cy * scale_h],
            [0.0,                0.0,               1.0]
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 1. 路径映射
        rgb_path = self.rgb_paths[idx]
        depth_path = rgb_path.replace('/rgb/', '/depth_z/')
        depth_z_path = rgb_path.replace('/rgb/', '/lidar_project/depth_z/')

        # 2. 加载图像并 Resize (注意 PIL resize 顺序是 (W, H))
        rgb_img = Image.open(rgb_path).convert('RGB').resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        depth_img = Image.open(depth_path).convert('F').resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        sparse_img = Image.open(depth_z_path).convert('F').resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        # 3. 转为 Tensor 并归一化深度 (AirSim PNG / 256.0 = Meters)
        rgb_tensor = TF.to_tensor(rgb_img) 
        depth_tensor = (torch.from_numpy(np.array(depth_img)) / 256.0).unsqueeze(0)
        sparse_tensor = (torch.from_numpy(np.array(sparse_img)) / 256.0).unsqueeze(0)
        
        # 清洗无效值 (inf, nan, 或超过 80 米的范围)
        depth_tensor = torch.nan_to_num(depth_tensor, 0, 0, 0)
        depth_tensor[depth_tensor >= 80.0] = 0
        sparse_tensor = torch.nan_to_num(sparse_tensor, 0, 0, 0)
        sparse_tensor[sparse_tensor >= 80.0] = 0

        Kcam = self.base_K.clone()

        # 4. 数据增强 (仅训练集)
        if self.split == 'train' and random.random() < self.p_augment:
            aug_ops = []

            # A. 随机水平翻转 (同步更新 cx)
            def flip_op(r, d, s):
                r_f = TF.hflip(r)
                d_f = TF.hflip(d)
                s_f = TF.hflip(s)
                Kcam[0, 2] = self.img_size[1] - 1 - Kcam[0, 2]
                return r_f, d_f, s_f
            aug_ops.append(lambda: flip_op(rgb_tensor, depth_tensor, sparse_tensor))

            # B. 颜色亮度对比度
            def color_op(r, d, s):
                r_aug = TF.adjust_brightness(r, random.uniform(0.8, 1.2))
                r_aug = TF.adjust_contrast(r_aug, random.uniform(0.8, 1.2))
                return r_aug, d, s
            aug_ops.append(lambda: color_op(rgb_tensor, depth_tensor, sparse_tensor))

            # C. 稀疏点随机重采样 (模拟不同密度的 LiDAR)
            def resample_op(r, d, s):
                num_p = random.choice([5, 10, 50, 100, 500, 1000])
                nonzero = torch.nonzero(s.squeeze())
                new_s = torch.zeros_like(s)
                if len(nonzero) > num_p:
                    sel = torch.randperm(len(nonzero))[:num_p]
                    coords = nonzero[sel]
                    for (h, w) in coords:
                        new_s[0, h, w] = s[0, h, w]
                    return r, d, new_s
                return r, d, s
            aug_ops.append(lambda: resample_op(rgb_tensor, depth_tensor, sparse_tensor))

            # D. 随机 Dropout (模拟丢失点)
            def dropout_op(r, d, s):
                mask = (torch.rand_like(s) < 0.1).float()
                return r, d, s * (1 - mask)
            aug_ops.append(lambda: dropout_op(rgb_tensor, depth_tensor, sparse_tensor))

            rgb_tensor, depth_tensor, sparse_tensor = random.choice(aug_ops)()

        # 5. 测试/验证集：固定采样点数
        elif self.fixed_num_points is not None:
            num_points = self.fixed_num_points
            nonzero_coords = torch.nonzero(sparse_tensor.squeeze())
            self.rng.seed(idx) # 保证每个样本的掩码固定
            new_sparse = torch.zeros_like(sparse_tensor)
            if num_points > 0 and len(nonzero_coords) > num_points:
                selected_indices = self.rng.choice(len(nonzero_coords), num_points, replace=False)
                selected_coords = nonzero_coords[selected_indices]
                for (h, w) in selected_coords:
                    new_sparse[0, h, w] = sparse_tensor[0, h, w]
                sparse_tensor = new_sparse
            elif num_points == 0:
                sparse_tensor = new_sparse

        # 6. 归一化 RGB
        rgb_norm = TF.normalize(rgb_tensor, self.norm_mean, self.norm_std)

        return rgb_norm, sparse_tensor, Kcam, depth_tensor