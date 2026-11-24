#!/usr/bin/env python3
"""One real A3 forward/backward/Adam/EMA update. Never writes official ckpts."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

import train_a3_portable as t


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    t.set_seed(t.SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=t.Final72RGBUMapDataset(cfg['paths']['train_csv'],cfg['paths']['raw_dir'],cfg['paths']['umap_dir'],training=True)
    loader=DataLoader(ds,batch_size=4,sampler=t.EpochShuffleSampler(ds,t.SEED),shuffle=False,drop_last=True,
        num_workers=int(cfg['train']['num_workers']),pin_memory=bool(cfg['train'].get('pin_memory',True)),
        persistent_workers=bool(cfg['train'].get('persistent_workers',False)) and int(cfg['train']['num_workers'])>0,
        worker_init_fn=t.seed_worker,generator=torch.Generator().manual_seed(t.SEED))
    if len(ds)!=4414:raise RuntimeError('Expected 4414 official training samples')
    model=t.RETFoundB15TaskQueryUMap4Head(checkpoint_path=t.resolve_ret_ckpt(cfg),image_size=384).to(device)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_trainable=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    if count!=1185253 or backbone_trainable!=0:
        raise RuntimeError('Unexpected A3 training parameters or unfrozen backbone')
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=t.BASE_LR,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
    ema=t.SimpleEMA(model,decay=t.EMA_DECAY)
    model.train()
    batch=next(iter(loader))
    img=batch['img'].to(device);u=batch['u_map'].float().to(device)
    gt=batch['gt'].float().to(device);mask=batch['gt_mask'].float().to(device)
    t.set_optimizer_lr(opt,0)
    opt.zero_grad(set_to_none=True)
    pred=model(img,u)
    loss,parts=t.four_task_loss(pred,gt,mask)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Loss is not finite')
    loss.backward()
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError('Frozen RETFound received gradients')
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm=math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads))
    if not math.isfinite(norm) or norm<=0:
        raise RuntimeError('Trainable A3 gradient norm invalid: %s'%norm)
    p=next(p for p in model.parameters() if p.requires_grad and p.grad is not None)
    before=p.detach().clone()
    opt.step()
    ema.update(model)
    changed=not torch.equal(before,p.detach())
    if not changed:raise RuntimeError('Adam did not update the checked trainable tensor')
    if not ema.shadow:raise RuntimeError('EMA shadow missing')
    print('[PASS] One real A3 train update + EMA; device=%s trainable=%d backbone_trainable=%d loss=%.6f grad_norm=%.6f'%(
        device,count,backbone_trainable,float(loss.detach().cpu()),norm),flush=True)
    print('[NOT A FULL TRAINING REPRODUCTION] No checkpoint written.',flush=True)


if __name__=='__main__':main()
