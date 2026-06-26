# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
from tqdm import tqdm
import hydra
import torch.distributed as dist
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

def train(run):
    for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)
        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)
        output = run.net(*datas[:-1])
        loss = run.criterion(output, datas[-1])
        sum(loss).backward()
        if run.clip:
            grad_norm = run.clip(run.net.parameters())
        run.optimizer.step()
        run.net_ema.update(run.net)
        if run.lr_iter:
            run.lr_scheduler.step()
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            run.writer.add_scalars("Loss", {f"{idx}": l.item() for idx, l in enumerate(loss)}, run.iter)
            if run.clip and (grad_norm is not None):
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)
        run.iter += 1
    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()


def test(run, iter=False):
    top1 = AverageMeter()
    net = run.net_ema.module
    best_metric_name = "best_metric_ema"
    legand = 'net_ema'
    net.eval()
    if not run.ddp or run.rank == 0:
        val_save_dir = os.path.join('results', run.cfg.name, 'val_vis')
        os.makedirs(val_save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
            datas = run.init_cuda(*datas)
            output = net(*datas[:-1])
            if isinstance(output, (list, tuple)):
                output = output[-1]
            prec1 = run.metric(output, datas[-1])
            if isinstance(prec1, (list, tuple)):
                prec1 = prec1[0]
            if run.ddp:
                dist.reduce(prec1, 0, dist.ReduceOp.AVG)
            top1.update(prec1.item())
            if not run.ddp or run.rank == 0:
                B = output.shape[0]
                for i in range(B):
                    # 计算当前 Epoch 内的全局样本索引
                    global_index = idx * B + i
                    
                    if global_index % 250 == 0:
                        file_name = f'epoch_{run.epoch}_{global_index}.png'
                        file_path = os.path.join(val_save_dir, file_name)

                        pred = output[i, 0].detach().cpu().numpy()
                        gt = datas[-1][i, 0].detach().cpu().numpy()

                        # 可视化准备
                        pred_vis = pred
                        gt_vis = gt
                        error = np.abs(pred - gt)

                        # 绘图
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # 统一颜色范围，保证对比清晰
                        max_val = max(pred_vis.max(), gt_vis.max()) + 1e-3

                        im0 = axs[0].imshow(pred_vis, cmap="plasma", vmin=0, vmax=max_val)
                        axs[0].set_title(f"Pred (min:{pred_vis.min():.2f})")
                        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

                        im1 = axs[1].imshow(gt_vis, cmap="plasma", vmin=0, vmax=max_val)
                        axs[1].set_title("GT")
                        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

                        im2 = axs[2].imshow(error, cmap="jet")
                        axs[2].set_title("Error")
                        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

                        for ax in axs:
                            ax.axis("off")

                        plt.tight_layout()
                        plt.savefig(file_path, dpi=150)
                        plt.close(fig)

    if iter:
        run.writer.add_scalars("RMSE_Iter", {legand: top1.avg}, run.iter)
    else:
        run.writer.add_scalars("RMSE", {legand: top1.avg}, run.epoch)
    if top1.avg < getattr(run, best_metric_name):
        setattr(run, best_metric_name, top1.avg)
        run.save_state()
        run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)
            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()
            run.save_state(filename='checkpoint_latest.pth')


if __name__ == '__main__':
    main()
