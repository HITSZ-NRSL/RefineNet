# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie

import torch
from tqdm import tqdm
import hydra
from PIL import Image
import os
from omegaconf import OmegaConf
from utils import *
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D

# --- 辅助函数：将图像裁剪为以高度为基准的正方形 ---
def crop_center_square(img):
    h, w = img.shape[:2]
    target_size = min(h, w) 
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    if img.ndim == 3:
        return img[start_h:start_h+target_size, start_w:start_w+target_size, :]
    else:
        return img[start_h:start_h+target_size, start_w:start_w+target_size]
# ----------------------------------------------------

# --- 辅助函数：渲染 3D 点云为图像帧 ---
def render_pc_to_image(depth, rgb, K, elevation=-20, azimuth=-90):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 过滤无效深度 (滤除极近噪点和极远背景)
    mask = (depth > 0.5) & (depth < 80.0)
    z = depth[mask]
    u = u[mask]
    v = v[mask]
    colors = rgb[mask]

    # 反投影 2D -> 3D
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 下采样加速渲染 (step=3)
    step = 3
    ax.scatter(x[::step], z[::step], -y[::step], c=colors[::step], s=1, marker='.')
    
    ax.view_init(elev=elevation, azim=azimuth)
    ax.axis('off')
    
    # 固定坐标轴范围，防止视频播放时画面忽大忽小
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 80)
    ax.set_zlim(-15, 15)
    
    fig.canvas.draw()
    frame_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_3d = frame_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame_3d
# ----------------------------------------------------

def test(run, mode='all', save=True, make_video=True):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()
    tops = [AverageMeter() for i in range(len(run.metric.metric_name))]
    
    if save:
        dir_path = f'results/{run.cfg.name}/{mode}'
        os.makedirs(dir_path, exist_ok=True)
        
    # --- 视频写入器初始化 ---
    video_writer = None
    if make_video:
        video_path = f'results/{run.cfg.name}/reconstruction.mp4'
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 左侧 400x798, 右侧 800x800 -> 整体画布 1200x800
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (1200, 800))
        video_frame_count = 0
        max_video_frames = 200 # 限制视频帧数，避免 Matplotlib 渲染时间过长

    fmt = FuncFormatter(lambda x, pos: f'{x:.1f}')

    with torch.no_grad():
        for idx, datas in enumerate(
                tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
            
            datas = run.init_cuda(*datas)
            # datas结构推断: 0:RGB, 1:Sparse, 2:Kcam, -1:GT
            output = net(*datas[:-1]) 
            
            if isinstance(output, (list, tuple)):
                output = output[-1]
                
            precs = run.metric(output, datas[-1])
            for prec, top in zip(precs, tops):
                top.update(prec.mean().detach().cpu().item())
            
            for i in range(output.shape[0]):
                index = idx * output.shape[0] + i
                
                # --- 数据提取与反归一化 ---
                rgb_raw = datas[0][i].detach().cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                rgb_denorm = rgb_raw * std + mean
                rgb_vis = np.clip(rgb_denorm, 0, 1) 

                sparse_raw = datas[1][i, 0].detach().cpu().numpy()
                gt_raw = datas[-1][i, 0].detach().cpu().numpy()
                pred_raw = output[i, 0].detach().cpu().numpy()
                
                # 假设数据集传递的第3个元素是相机内参矩阵 K
                K_matrix = datas[2][i].detach().cpu().numpy()

                # ================= 视频生成逻辑 =================
                if make_video and video_writer is not None and video_frame_count < max_video_frames:
                    # 1. 准备左侧面板图 (RGB, Sparse, Pred)
                    lw, lh = 400, 266 # 单张图尺寸
                    
                    rgb_left = cv2.resize((rgb_vis * 255).astype(np.uint8), (lw, lh))
                    
                    # 伪彩色映射 (Plasma)
                    vmax = np.max(pred_raw) if np.max(pred_raw) > 0 else 50.0
                    sparse_vis_img = plt.cm.plasma(np.clip(sparse_raw / vmax, 0, 1))[:,:,:3]
                    sparse_left = cv2.resize((sparse_vis_img * 255).astype(np.uint8), (lw, lh))
                    
                    pred_vis_img = plt.cm.plasma(np.clip(pred_raw / vmax, 0, 1))[:,:,:3]
                    pred_left = cv2.resize((pred_vis_img * 255).astype(np.uint8), (lw, lh))
                    
                    # 添加标签
                    cv2.putText(rgb_left, "Input RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(sparse_left, "Input Sparse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(pred_left, "Pred Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    left_panel = np.vstack([rgb_left, sparse_left, pred_left]) # 尺寸: 400 x 798
                    
                    # 补齐底部的 2 个像素使其达到 800 高度
                    left_panel = cv2.copyMakeBorder(left_panel, 0, 2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                    # 2. 生成右侧 3D 面板
                    # 加入缓慢旋转效果 (方位角随帧数改变)
                    azim = -90 + (video_frame_count * 0.5) 
                    right_panel = render_pc_to_image(pred_raw, rgb_vis, K_matrix, elevation=-30, azimuth=azim)
                    right_panel = cv2.resize(right_panel, (800, 800))
                    
                    # 3. 拼接并写入视频
                    full_frame = np.hstack([left_panel, right_panel])
                    video_writer.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))
                    video_frame_count += 1
                # ===============================================

                # ================= 原有的保存图像逻辑 =================
                if save and idx % 100 == 0 and i == 0: 
                    file_path = os.path.join(dir_path, f'{index:010d}.png')
                    # ... [此处保留你原本 crop_center_square 和 GridSpec 画图的代码，为节省篇幅折叠] ...
                    # 注意：将你原来 if save 内部的图像裁剪、膨胀和 GridSpec 绘图代码直接放在这里即可。
                    pass
                # ==================================================

    if make_video and video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {video_path}")

    logs = ""
    for name, top in zip(run.metric.metric_name, tops):
        logs += f" {name}:{top.avg:.7f} "
    run.ddp_log(logs, always=True)

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        test(run,
             save=OmegaConf.select(cfg, 'save', default=True),
             make_video=OmegaConf.select(cfg, 'make_video', default=False))

if __name__ == '__main__':
    main()