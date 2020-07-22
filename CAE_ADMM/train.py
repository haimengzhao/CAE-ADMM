import argparse
import os
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model import CAEP
from utils import BSDS500Crop128, Kodak, compute_bpp, save_kodak_img, compute_psnr
import pytorch_msssim


num_resblocks = 15
rho = 1e-1
pruning_ratio = 0.9


def train(args):

    print('Number of GPUs available: ' + str(torch.cuda.device_count()))
    model = nn.DataParallel(CAEP(num_resblocks).cuda())
    print('Done Setup Model.')

    dataset = BSDS500Crop128(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers)
    testset = Kodak(args.testset_path)
    testloader = DataLoader(
        testset,
        batch_size=testset.__len__(),
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
    Z = torch.zeros(16,32,32).cuda()
    U = torch.zeros(16,32,32).cuda()
    Z.requires_grad = False
    U.requires_grad = False

    if args.load != '':
        pretrained_state_dict = torch.load(f"./ckpt/{args.load}/model.state")
        current_state_dict = model.state_dict()
        current_state_dict.update(pretrained_state_dict)
        model.load_state_dict(current_state_dict)
        # Z = torch.load(f"./ckpt/{args.load}/Z.state")
        # U = torch.load(f"./ckpt/{args.load}/U.state")
        if args.load == args.exp_name:
            optimizer.load_state_dict(torch.load(f"./ckpt/{args.load}/opt.state"))
            scheduler.load_state_dict(torch.load(f"./ckpt/{args.load}/lr.state"))
        print('Model Params Loaded.')

    model.train()

    for ei in range(args.res_epoch + 1, args.res_epoch + args.num_epochs + 1):
        # train
        train_loss = 0
        train_ssim = 0
        train_msssim = 0
        train_psnr = 0
        train_peanalty = 0
        train_bpp = 0
        avg_c = torch.zeros(16,32,32).cuda()
        avg_c.requires_grad = False

        for bi, crop in enumerate(dataloader):
            x = crop.cuda()
            y, c = model(x)

            psnr = compute_psnr(x, y)
            mse = MSE(y, x)
            ssim = SSIM(x, y)
            msssim = MSSSIM(x, y)

            mix = 1000 * (1 - msssim) + 1000 * (1 - ssim) + 1e4 * mse + (45 - psnr)
            # ADMM Step 1
            peanalty = rho / 2 * torch.norm(c - Z + U, 2)
            bpp = compute_bpp(c, x.shape[0])

            avg_c += torch.mean(c.detach() / (len(dataloader) * args.admm_every), dim=0)

            loss = mix + peanalty
            if ei == 1 and args.load != args.exp_name:
                loss = 1e5 * mse  # warm up

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[%3d/%3d][%5d/%5d] Loss: %f, SSIM: %f, MSSSIM: %f, PSNR: %f, Norm of Code: %f, BPP: %2f' %
                  (ei, args.num_epochs + args.res_epoch, bi, len(dataloader), loss, ssim, msssim, psnr, peanalty, bpp))
            writer.add_scalar('batch_train/loss', loss, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/ssim', ssim, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/msssim', msssim, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/psnr', psnr, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/norm', peanalty, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/bpp', bpp, ei * len(dataloader) + bi)

            train_loss += loss.item() / len(dataloader)
            train_ssim += ssim.item() / len(dataloader)
            train_msssim += msssim.item() / len(dataloader)
            train_psnr += psnr.item() / len(dataloader)
            train_peanalty += peanalty.item() / len(dataloader)
            train_bpp += bpp / len(dataloader)

        writer.add_scalar('epoch_train/loss', train_loss, ei)
        writer.add_scalar('epoch_train/ssim', train_ssim, ei)
        writer.add_scalar('epoch_train/msssim', train_msssim, ei)
        writer.add_scalar('epoch_train/psnr', train_psnr, ei)
        writer.add_scalar('epoch_train/norm', train_peanalty, ei)
        writer.add_scalar('epoch_train/bpp', train_bpp, ei)

        if ei % args.admm_every == args.admm_every - 1:
            # ADMM Step 2
            Z = (avg_c + U).masked_fill_(
                (torch.Tensor(np.argsort((avg_c + U).data.cpu().numpy(), axis=None))
                 >= int((1 - pruning_ratio) * 16 * 32 * 32)).view(16, 32, 32).cuda(),
                value=0)
            # ADMM Step 3
            U += avg_c - Z

        # test
        model.eval()
        val_loss = 0
        val_ssim = 0
        val_msssim = 0
        val_psnr = 0
        val_peanalty = 0
        val_bpp = 0
        for bi, (img, patches, _) in enumerate(testloader):
            avg_loss = 0
            avg_ssim = 0
            avg_msssim = 0
            avg_psnr = 0
            avg_peanalty = 0
            avg_bpp = 0
            for i in range(6):
                for j in range(4):
                    x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
                    y, c = model(x)

                    psnr = compute_psnr(x, y)
                    mse = MSE(y, x)
                    ssim = SSIM(x, y)
                    msssim = MSSSIM(x, y)

                    mix = 1000 * (1 - msssim) + 1000 * (1 - ssim) + 1e4 * mse + (45 - psnr)

                    peanalty = rho / 2 * torch.norm(c - Z + U, 2)
                    bpp = compute_bpp(c, x.shape[0])
                    loss = mix + peanalty

                    avg_loss += loss.item() / 24
                    avg_ssim += ssim.item() / 24
                    avg_msssim += msssim.item() / 24
                    avg_psnr += psnr.item() / 24
                    avg_peanalty += peanalty.item() / 24
                    avg_bpp += bpp / 24
            
            save_kodak_img(model, img, 0, patches, writer, ei)
            save_kodak_img(model, img, 10, patches, writer, ei)
            save_kodak_img(model, img, 20, patches, writer, ei)
            
            val_loss += avg_loss
            val_ssim += avg_ssim
            val_msssim += avg_msssim
            val_psnr += avg_psnr
            val_peanalty += avg_peanalty
            val_bpp += avg_bpp
        print('*Kodak: [%3d/%3d] Loss: %f, SSIM: %f, MSSSIM: %f, Norm of Code: %f, BPP: %.2f' %
              (ei, args.num_epochs + args.res_epoch, val_loss, val_ssim, val_msssim, val_peanalty, val_bpp))

        # bz = call('tar -jcvf ./code/code.tar.bz ./code', shell=True)
        # total_code_size = os.stat('./code/code.tar.bz').st_size
        # total_bpp = total_code_size * 8 / 24 / 768 / 512

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
            torch.save(model.state_dict(), f"./ckpt/{args.exp_name}/model.state")
            torch.save(optimizer.state_dict(), f"./ckpt/{args.exp_name}/opt.state")
            torch.save(scheduler.state_dict(), f"./ckpt/{args.exp_name}/lr.state")
            torch.save(Z, f"./ckpt/{args.exp_name}/Z.state")
            torch.save(U, f"./ckpt/{args.exp_name}/U.state")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--res_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--out_every', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../data/BSDS500')
    parser.add_argument('--testset_path', type=str, default='../data/Kodak')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--admm_every', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(f"./output", exist_ok=True)
    os.makedirs(f"./ckpt/{args.exp_name}", exist_ok=True)
    os.makedirs(f"./code", exist_ok=True)

    train(args)
