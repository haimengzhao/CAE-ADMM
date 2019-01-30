import argparse
import os
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model import CAEP
from utils import GeneralDS, BSDS500Crop128, Kodak, compute_bpp, save_kodak_img, compute_psnr
import pytorch_msssim
from tqdm import tqdm
import time

num_resblocks = 15
rho = 1e-1
pruning_ratio = 0.9


def train(args):

    print('Number of GPUs available: ' + str(torch.cuda.device_count()))
    model = nn.DataParallel(CAEP(num_resblocks,16).cuda())
    print('Done Setup Model.')

    dataset = BSDS500Crop128(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=args.shuffle,
        num_workers=args.num_workers)
    testset = GeneralDS('./mixed')#Kodak(args.testset_path)
    testloader = DataLoader(
        testset,
        batch_size=8,#testset.__len__(),
        num_workers=args.num_workers)
    print(f"Done Setup Training DataLoader: {len(dataloader)} batches of size {args.batch_size}")
    print(f"Done Setup Testing DataLoader: {len(testset)} Images")

    MSE = nn.MSELoss()
    SSIM = pytorch_msssim.SSIM().cuda()
    MSSSIM = pytorch_msssim.MSSSIM().cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-10
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
    )

    writer = SummaryWriter(log_dir=f'TBXLog/{args.exp_name}')

    # ADMM variables
    Z = torch.zeros(16,16,16).cuda()
    U = torch.zeros(16,16,16).cuda()
    Z.requires_grad = False
    U.requires_grad = False

    if args.load != '':
        pretrained_state_dict = torch.load(f"./chkpt/{args.load}/model.state")
        current_state_dict = model.state_dict()
        current_state_dict.update(pretrained_state_dict)
        model.load_state_dict(current_state_dict)
        # Z = torch.load(f"./chkpt/{args.load}/Z.state")
        # U = torch.load(f"./chkpt/{args.load}/U.state")
        if args.load == args.exp_name:
            optimizer.load_state_dict(torch.load(f"./chkpt/{args.load}/opt.state"))
            scheduler.load_state_dict(torch.load(f"./chkpt/{args.load}/lr.state"))
        print('Model Params Loaded.')

    model.train()
    
    for ei in range(args.res_epoch + 1, args.res_epoch + args.num_epochs + 1):
        # test
        model.eval()
        val_loss = []
        val_ssim = []
        val_msssim = []
        val_psnr = []
        val_penalty = []
        val_bpp = []

        times = []
        for bi, (img, patches, _) in tqdm(enumerate(testloader)):
            avg_loss = 0
            avg_ssim = 0
            avg_msssim = 0
            avg_psnr = 0
            avg_penalty = 0
            avg_bpp = 0
            time_im = 0
            for i in range(6):
                for j in range(4):
                    start_time = time.time()
                    x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
                    y, c = model(x)
                    time_im += time.time() - start_time

                    psnr = compute_psnr(x, y)
                    mse = MSE(y, x)
                    ssim = SSIM(x, y)
                    msssim = MSSSIM(x, y)

                    mix = 1000 * (1 - msssim) + 1000 * (1 - ssim) + 1e4 * mse + (45 - psnr)

                    peanalty = rho / 2 * torch.norm(c - Z + U, 2)
                    bpp = compute_bpp(c, x.shape[0], f'Kodak_patches_{i}_{j}', save=True)
                    loss = mix + peanalty

                    avg_loss += loss.item() / 24
                    avg_ssim += ssim.item() / 24
                    avg_msssim += msssim.item() / 24
                    avg_psnr += psnr.item() / 24
                    avg_penalty += peanalty.item() / 24
                    avg_bpp += bpp / 24
            
            #save_kodak_img(model, img, 0, patches, writer, ei)
            #save_kodak_img(model, img, 10, patches, writer, ei)
            #save_kodak_img(model, img, 20, patches, writer, ei)
            
            val_loss.append( avg_loss )
            val_ssim.append( avg_ssim )
            val_msssim.append( avg_msssim )
            val_psnr.append( avg_psnr )
            val_penalty.append( avg_penalty )
            val_bpp.append( avg_bpp )
            times.append(time_im/img.size()[0])
            #break
        
        std_loss = np.array(val_loss).std()
        std_ssim = np.array(val_ssim).std()
        std_msssim = np.array(val_msssim).std()
        std_penalty = np.array(val_penalty).std()
        std_bpp = np.array(val_bpp).std()

        val_loss = np.array(val_loss).mean()
        val_ssim = np.array(val_ssim).mean()
        val_msssim = np.array(val_msssim).mean()
        val_penalty = np.array(val_penalty).mean() 
        val_bpp = np.array(val_bpp).mean()
        print('*mixed: [%3d/%3d] Loss: %f, SSIM: %f, MSSSIM: %f, Norm of Code: %f, BPP: %.3f' %
              (ei, args.num_epochs + args.res_epoch, val_loss, val_ssim, val_msssim, val_penalty, val_bpp))
        
        print('*  std: [%3d/%3d] Loss: %f, SSIM: %f, MSSSIM: %f, Norm of Code: %f, BPP: %.3f' %
              (ei, args.num_epochs + args.res_epoch, std_loss, std_ssim, std_msssim, std_penalty, std_bpp))
        print('avg time: ', np.array(times).mean())
        print('std: ', np.array(times).std())
        # bz = call('tar -jcvf ./code/code.tar.bz ./code', shell=True)
        # total_code_size = os.stat('./code/code.tar.bz').st_size
        # total_bpp = total_code_size * 8 / 24 / 768 / 512
        break

        writer.add_scalar('test/loss', val_loss, ei)
        writer.add_scalar('test/ssim', val_ssim, ei)
        writer.add_scalar('test/msssim', val_msssim, ei)
        writer.add_scalar('test/psnr', val_psnr, ei)
        writer.add_scalar('test/norm', val_peanalty, ei)
        writer.add_scalar('test/bpp', val_bpp, ei)
        # writer.add_scalar('test/total_bpp', total_bpp, ei)
        model.train()

        scheduler.step(train_loss)

        # save model
        if ei % args.save_every == args.save_every - 1:
            torch.save(model.state_dict(), f"./chkpt/{args.exp_name}/model.state")
            torch.save(optimizer.state_dict(), f"./chkpt/{args.exp_name}/opt.state")
            torch.save(scheduler.state_dict(), f"./chkpt/{args.exp_name}/lr.state")
            torch.save(Z, f"./chkpt/{args.exp_name}/Z.state")
            torch.save(U, f"./chkpt/{args.exp_name}/U.state")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--res_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--load', type=str, default='bpp_05')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--out_every', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--testset_path', type=str, default='./Kodak')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--admm_every', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(f"./output", exist_ok=True)
    os.makedirs(f"./chkpt/{args.exp_name}", exist_ok=True)
    os.makedirs(f"./code", exist_ok=True)

    train(args)
