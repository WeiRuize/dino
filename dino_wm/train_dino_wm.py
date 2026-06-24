import os
import numpy as np
import torch
import torch.distributed as dist
import random
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.optim import AdamW
from torch import nn
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm

from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, load_action_bounds
import libero_config as C


def setup_distributed():
    """Init DDP when launched via torchrun, else fall back to single-GPU.

    Returns (rank, world_size, local_rank, device, is_distributed). Plain
    `python train_dino_wm.py` (no torchrun env vars) runs as a single process
    on cuda:0 exactly as before.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, f"cuda:{local_rank}", True
    return 0, 1, 0, "cuda:0", False


dino = C.load_dino()


transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(224),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


DINO_transform = transforms.Compose([           
                            transforms.Resize(224),
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )

if __name__ == "__main__":
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    is_main = rank == 0

    # Only rank 0 logs to wandb to avoid duplicate runs.
    if is_main:
        wandb.init(project="dino-WM", name="WM")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


    BS = 16
    BL= 4
    EVAL_H = 16
    H = 3

    # LIBERO action normalization bounds (written by libero_to_dataset.py).
    load_action_bounds(C.ACTION_STATS_PATH)

    hdf5_file = C.CONSOLIDATED_TRAIN

    # Split one consolidated file by fraction (train = trailing TRAIN_FRAC,
    # test = leading rest), matching latent-safety's RSSM train/val split.
    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', train_frac=C.TRAIN_FRAC)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, BL, split='test', train_frac=C.TRAIN_FRAC)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, 32, split='test', train_frac=C.TRAIN_FRAC)

    # Under DDP each rank trains on a disjoint shard of the data (DistributedSampler);
    # single-GPU keeps the original shuffle=True behaviour.
    if is_distributed:
        train_sampler = DistributedSampler(expert_data, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None
    expert_loader = iter(DataLoader(expert_data, batch_size=BS,
                                    sampler=train_sampler, shuffle=(train_sampler is None),
                                    num_workers=C.NUM_WORKERS, pin_memory=True))
    # Eval / imagine run on rank 0 only, so they stay plain non-distributed loaders.
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    # Decoder is only used for the rank-0 visualisation/eval block.
    if is_main:
        decoder = VQVAE().to(device)
        decoder.load_state_dict(torch.load('checkpoints/testing_decoder.pth'))
        decoder.eval()

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL-1,
        dropout=0.1,
        device=device  # build submodules (incl. frozen DINO) on this rank's GPU
    ).to(device)
    transition.train()

    if is_distributed:
        # static_graph=True is required here: the step runs two forwards (teacher
        # forcing + autoregressive) that reuse the same params before one backward,
        # and failure_head is computed but unused in WM training. static_graph
        # handles both reused and statically-unused params (the frozen DINO encoder
        # has requires_grad=False so DDP ignores it).
        transition = DDP(transition, device_ids=[local_rank], output_device=local_rank,
                         static_graph=True)
    # Underlying module for submodule access, eval forward, and checkpointing.
    net = transition.module if is_distributed else transition

    # Forward pass
    optimizer = AdamW([
        {'params': net.transformer.parameters(), 'lr': 5e-5},
        {'params': net.state_head.parameters(), 'lr': 5e-5},
        {'params': net.front_head.parameters(), 'lr': 5e-5},
        {'params': net.wrist_head.parameters(), 'lr': 5e-5},
        {'params': net.action_encoder.parameters(), 'lr': 5e-4},
        {'params': [net.pos_embedding], 'lr': 5e-4},
        {'params': [net.temp_embedding], 'lr': 5e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_iter = 100000

    for i in tqdm(range(train_iter), desc="Training", unit="iter", disable=not is_main):
        if i % len(expert_loader) == 0:
            # Reshuffle each pass; under DDP set_epoch gives each rank a new shard order.
            if is_distributed:
                train_sampler.set_epoch(i // len(expert_loader))
            expert_loader = iter(DataLoader(expert_data, batch_size=BS,
                                            sampler=train_sampler, shuffle=(train_sampler is None),
                                            num_workers=C.NUM_WORKERS, pin_memory=True))

        data = next(expert_loader)


        data1 = data['cam_zed_embd'].to(device)
        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        data2 =  data['cam_rs_embd'].to(device)
        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        inputs_states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(data_acs, device)
        acs = norm_acs[:, :-1]

        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)
            im1_loss_tf = nn.MSELoss()(pred1, output1)
            im2_loss_tf = nn.MSELoss()(pred2, output2)
            state_loss_tf = nn.MSELoss()(pred_state, output_state)
            loss_tf = im1_loss_tf + im2_loss_tf + state_loss_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            detach_pred1 = pred1
            detach_pred2 = pred2
            detach_pred_state = pred_state.detach()
            inputs1_ar = torch.cat([data1[:, [0]], detach_pred1[:, [0]]], dim=1)
            inputs2_ar = torch.cat([data2[:, [0]], detach_pred2[:, [0]]], dim=1)
            states_ar = torch.cat([data_state[:,[0]], detach_pred_state[:, [0]]], dim=1)
            acs_ar = norm_acs[:, [0,1]]

            pred1_ar, pred2_ar, pred_state_ar, _ = transition(inputs1_ar, inputs2_ar, states_ar, acs_ar)
            output1_ar = data1[:, 2]
            output2_ar = data2[:, 2]
            output_state_ar = data_state[:, 2]
            im1_loss_ar = nn.MSELoss()(pred1_ar[:,1], output1_ar)
            im2_loss_ar = nn.MSELoss()(pred2_ar[:,1], output2_ar)
            state_loss_ar = nn.MSELoss()(pred_state_ar[:,1], output_state_ar)
            loss_ar = im1_loss_ar + im2_loss_ar + state_loss_ar       

        loss = loss_tf + loss_ar*0.5
        #loss = loss_tf

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        if is_main:
            print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, front Loss: {im1_loss_tf.item():.4f}, wrist Loss: {im2_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}", end='', flush=True)
            wandb.log({'train_loss': loss_tf, "train_loss_ar": loss_ar})
        # eval (rank 0 only)
        if is_main and (i) % 1000 == 0:
            iters.append(i)
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))
            eval_data = next(expert_loader_imagine)
            net.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_embd'].to(device)
                inputs1 = eval_data1[[0], :H].to(device)

                eval_data2 =  eval_data['cam_rs_embd'].to(device)
                inputs2 = eval_data2[[0], :H].to(device)
                
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(all_acs, device)
                
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(acs, device)

                inputs_states = eval_data['state'][[0],:H].to(device)
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                for k in range(EVAL_H-H):
                    pred1, pred2, pred_state, _ = net(inputs1, inputs2, inputs_states, acs)

                    pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)#.squeeze()
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)

                    
                    im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
                    im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
                    
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    states = torch.cat([inputs_states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                    
                gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-2)
                vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                vid = rearrange(vid, "t h w c -> t c h w")
                wandb.log({"video": wandb.Video(vid, fps=20, format='mp4')})
                
                # done logging video

    
                expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
                eval_data = next(expert_loader_eval)
                data1 = eval_data['cam_zed_embd'].to(device)
                data2 =  eval_data['cam_rs_embd'].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = eval_data['state'].to(device)
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data['action'].to(device)
                data_acs = normalize_acs(data_acs, device)
                acs = data_acs[:, :-1]
                pred1, pred2, pred_state, _ = net(inputs1, inputs2, states, acs)


                pred_latent = torch.cat([pred1[:,[H-1]], pred2[:,[H-1]]], dim=0)
                pred_ims, _ = decoder(pred_latent)
                pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                pred_im1 = pred_im1[0].permute(1,2,0).detach().cpu().numpy()
                pred_im2 = pred_im2[0].permute(1,2,0).detach().cpu().numpy()
                im1 = eval_data['agentview_image'][0, H].numpy()
                im2 = eval_data['robot0_eye_in_hand_image'][0, H].numpy()
                im1_loss = nn.MSELoss()(pred1, output1)
                im2_loss = nn.MSELoss()(pred2, output2)
                state_loss = nn.MSELoss()(pred_state, output_state)
                loss = im1_loss + im2_loss + state_loss
            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {im1_loss.item():.4f}, wrist Loss: {im2_loss.item():.4f}, state Loss: {state_loss.item():.4f}")

            torch.save(net.state_dict(), f'checkpoints/testing_iter{i}.pth')

            if loss < best_eval:
                best_eval = loss
                torch.save(net.state_dict(), 'checkpoints/best_testing.pth')

            net.train()
            wandb.log({'eval_loss': loss.item(), 'front_loss': im1_loss.item(), 'wrist_loss': im2_loss.item(), 'state_loss': state_loss.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(im1), 'wrist': wandb.Image(im2)})


    if is_main:
        plt.legend()
        plt.savefig('training curve.png')
    if is_distributed:
        dist.destroy_process_group()