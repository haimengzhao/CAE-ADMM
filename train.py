import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model import CAEP
from utils import BSDS500Crop128, Kodak, compute_bpp, save_kodak_img
import pytorch_msssim


num_resblocks = 15
rho = 1e-7


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
    print(f"Done Setup Testing DataLoader: {len(testset)}")

    MSE = nn.MSELoss()
    SSIM = pytorch_msssim.SSIM().cuda()
    MSSSIM = pytorch_msssim.MSSSIM().cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-7
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
    )

    writer = SummaryWriter(log_dir='TBXLog')

    if args.load:
        model.load_state_dict(torch.load(f"./chkpt/model.state"))
        optimizer.load_state_dict(torch.load(f"./chkpt/opt.state"))
        scheduler.load_state_dict(torch.load(f"./chkpt/lr.state"))
        print('Model Params Loaded.')

    model.train()

    for ei in range(args.res_epoch + 1, args.res_epoch + args.num_epochs + 1):
        # train
        train_loss = 0
        train_ssim = 0
        train_msssim = 0
        train_peanalty = 0
        train_bpp = 0
        for bi, crop in enumerate(dataloader):
            x = crop.cuda()
            y, c = model(x)

            mse = MSE(y, x)
            ssim = SSIM(x, y)
            msssim = MSSSIM(x, y)
            peanalty = rho / 2 * torch.norm(c, 2)
            bpp = compute_bpp(c, x.shape[0], 'train', save=False)

            loss = mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[%3d/%3d][%5d/%5d] Loss: %f, SSIM: %f, MSSSIM: %f, Norm of Code: %f, BPP: %2f' %
                  (ei, args.num_epochs, bi, len(dataloader), loss, ssim, msssim, peanalty, bpp))
            writer.add_scalar('batch_train/loss', loss, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/ssim', ssim, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/msssim', msssim, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/norm', peanalty, ei * len(dataloader) + bi)
            writer.add_scalar('batch_train/bpp', bpp, ei * len(dataloader) + bi)

            train_loss += loss.item() / len(dataloader)
            train_ssim += ssim.item() / len(dataloader)
            train_msssim += msssim.item() / len(dataloader)
            train_peanalty += peanalty.item() / len(dataloader)
            train_bpp += bpp / len(dataloader)

        writer.add_scalar('epoch_train/loss', train_loss, ei)
        writer.add_scalar('epoch_train/ssim', train_ssim, ei)
        writer.add_scalar('epoch_train/msssim', train_msssim, ei)
        writer.add_scalar('epoch_train/norm', train_peanalty, ei)
        writer.add_scalar('epoch_train/bpp', train_bpp, ei)

        # test
        model.eval()
        val_loss = 0
        val_ssim = 0
        val_msssim = 0
        val_peanalty = 0
        val_bpp = 0
        for bi, (img, patches, _) in enumerate(testloader):
            avg_loss = 0
            avg_ssim = 0
            avg_msssim = 0
            avg_peanalty = 0
            avg_bpp = 0
            for i in range(6):
                for j in range(4):
                    x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
                    y, c = model(x)

                    mse = MSE(y, x)
                    ssim = SSIM(x, y)
                    msssim = MSSSIM(x, y)
                    peanalty = rho / 2 * torch.norm(c, 2)
                    bpp = compute_bpp(c, x.shape[0], f'Kodak_{i}_{j}', save=True)
                    loss = mse

                    avg_loss += loss.item() / 24
                    avg_ssim += ssim.item() / 24
                    avg_msssim += msssim.item() / 24
                    avg_peanalty += peanalty.item() / 24
                    avg_bpp += bpp / 24

            print('*[%3d/%3d]The average SSIM in Kodak: %f, Loss: %f, BPP: %f' %
                  (ei, args.num_epochs, avg_ssim, avg_loss, avg_bpp))
            
            save_kodak_img(model, img, 0, patches, writer, ei)
            save_kodak_img(model, img, 10, patches, writer, ei)
            save_kodak_img(model, img, 20, patches, writer, ei)
            
            val_loss += avg_loss
            val_ssim += avg_ssim
            val_msssim += avg_msssim
            val_peanalty += avg_peanalty
            val_bpp += avg_bpp

        writer.add_scalar('test/loss', val_loss, ei)
        writer.add_scalar('test/ssim', val_ssim, ei)
        writer.add_scalar('test/msssim', val_msssim, ei)
        writer.add_scalar('test/norm', val_peanalty, ei)
        writer.add_scalar('test/bpp', val_bpp, ei)
        model.train()

        scheduler.step(train_loss)

        # save model
        if ei % args.save_every == args.save_every - 1:
            torch.save(model.state_dict(), f"./chkpt/model.state")
            torch.save(optimizer.state_dict(), f"./chkpt/opt.state")
            torch.save(scheduler.state_dict(), f"./chkpt/lr.state")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--res_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--out_every', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--testset_path', type=str, default='../Kodak')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(f"./output", exist_ok=True)
    os.makedirs(f"./chkpt", exist_ok=True)
    os.makedirs(f"./code", exist_ok=True)

    train(args)
