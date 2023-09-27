import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from basemodels import cusResNet152
from Fitz17k import Fitz17k
from model import resnet152
from rich.console import Console
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from utils import *


class SPD_Loss(nn.Module):
    def __init__(self) -> None:
        super(SPD_Loss, self).__init__()
    
    def forward(self, preds, attrs):
        
        spd = 0
        predictions = torch.zeros(size=(9,2))
        
        for i in range(len(preds)):
            predictions[preds[i]][attrs[i]] += 1
        
        n_attr_1 = torch.nonzero(attrs).shape[0]
        n_attr_0 = attrs.shape[0] - n_attr_1
        
        for i in range(9):
            spd += torch.pow(input=(predictions[i][0] / n_attr_0 - predictions[i][1] / n_attr_1), exponent=2)
        
        return spd

def train(net, criterion, train_loader, valid_loader, optimizer, max_epoch, valid_interval=10):
    # inits
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_acc = 0
    spd = SPD_Loss()
  
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeRemainingColumn(),
                  TimeElapsedColumn(), console=console) as progress:
        epoch_tqdm = progress.add_task(description="epoch progress", total=max_epoch)
        train_tqdm = progress.add_task(description="train progress", total=len(train_loader))
        valid_tqdm = progress.add_task(description="valid progress", total=len(valid_loader))
        # start iteration
        for iteration in range(max_epoch):
            console.rule('epoch {}'.format(iteration))     
                   
            # train
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()          
            epoch_loss_ce = AverageMeter()
            epoch_loss_spd = AverageMeter()
            net.train()
            for _, image, label, sensitive_attribute in train_loader:
                image, label, sensitive_attribute = image.cuda(), label.cuda(), sensitive_attribute.cuda()
                
                task_0_idx = (sensitive_attribute == 0).nonzero(as_tuple=True)
                task_1_idx = (sensitive_attribute == 1).nonzero(as_tuple=True)
                
                image_0, label_0, sensitive_attribute_0 = image[task_0_idx], label[task_0_idx], sensitive_attribute[task_0_idx]
                image_1, label_1, sensitive_attribute_1 = image[task_1_idx], label[task_1_idx], sensitive_attribute[task_1_idx]
                
                label_0, label_1 = label_0.squeeze(dim=1), label_1.squeeze(dim=1)
                
                # compute loss on group 0

                output, _ = net(image_0, task_idx=0)
                loss_0 = criterion(output, label_0.long())
                preds_0 = output.argmax(dim=1)

                output, _ = net(image_1, task_idx=1)
                loss_1 = criterion(output, label_1.long())
                preds_1 = output.argmax(dim=1)
                
                # combine
                loss_spd = spd(torch.cat([preds_0, preds_1]), torch.cat([sensitive_attribute_0, sensitive_attribute_1]))
                loss = loss_0 + loss_1 + loss_spd
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                
                epoch_loss.update(loss.item())
                epoch_loss_ce.update(loss_0.item() + loss_1.item())
                epoch_loss_spd.update(loss_spd.item())
                epoch_acc.update((sum(preds_0 == label_0.squeeze()) / label_0.shape[0]).item())
                if label_1.shape[0] != 0:
                    epoch_acc.update((sum(preds_1 == label_1.squeeze()) / label_1.shape[0]).item())
                
                progress.advance(train_tqdm, advance=1)
            
            console.log('train acc = {:.4f}\ttrain loss = {:.4f}\ttrain ce loss = {:.4f}\ttrain spd loss = {:.4f}'.format(epoch_acc.avg, epoch_loss.avg, epoch_loss_ce.avg, epoch_loss_spd.avg))
            train_losses.append(epoch_loss.avg)
            train_accs.append(epoch_acc.avg)
            progress.reset(train_tqdm)
            
            scheduler.step()
            
            # valid
            if iteration % valid_interval == 0:
                epoch_loss = AverageMeter()
                epoch_acc = AverageMeter()
                net.eval()
                for _, image, label, sensitive_attribute in valid_loader:
                    image, label, sensitive_attribute = image.cuda(), label.cuda(), sensitive_attribute.cuda()
                    
                    task_0_idx = (sensitive_attribute == 0).nonzero(as_tuple=True)
                    task_1_idx = (sensitive_attribute == 1).nonzero(as_tuple=True)
                    
                    image_0, label_0 = image[task_0_idx], label[task_0_idx]
                    image_1, label_1 = image[task_1_idx], label[task_1_idx]
                    
                    label_0, label_1 = label_0.squeeze(dim=1), label_1.squeeze(dim=1)
        
                    with torch.no_grad():
                        # compute loss on group 0
                        output, _ = net(image_0, task_idx=0)
                        loss_0 = criterion(output, label_0.long())
                        preds_0 = output.argmax(dim=1)

                        # compute loss on group 1
                        output, _ = net(image_1, task_idx=1)
                        loss_1 = criterion(output, label_1.long())
                        preds_1 = output.argmax(dim=1)
                    
                    # combine
                    loss = loss_0 + loss_1 
                                    
                    epoch_loss.update(loss.item())
                    epoch_acc.update((sum(preds_0 == label_0.squeeze()) / label_0.shape[0]).item())
                    
                    if label_1.shape[0] != 0:
                        epoch_acc.update((sum(preds_1 == label_1.squeeze()) / label_1.shape[0]).item())
                    
                    progress.advance(valid_tqdm, advance=1)
                
                console.log('valid acc = {:.4f}\tvalid loss = {:.4f}'.format(epoch_acc.avg, epoch_loss.avg))
                valid_losses.append(epoch_loss.avg)
                valid_accs.append(epoch_acc.avg)
                
                if epoch_acc.avg > best_acc:
                    best_acc = epoch_acc.avg
                    console.log('save model with acc: {:.4f}'.format(best_acc))
                    save_best_model(net, args.exp_id, args.rand_seed)
                
                progress.reset(valid_tqdm)
 
            progress.advance(epoch_tqdm, advance=1)
            
    return train_accs, train_losses, valid_accs, valid_losses
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train settings')
    parser.add_argument('--exp_id', default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--rand_seed', type=str, default=0)
        
    args = parser.parse_args()
    
    # logs
    console = Console()
    console.log('baseline model with backbone resnet-152')
    console.log(args)
    
    # paths
    ann_train = pd.read_csv('./dataset/Fitzpatrick-17k/processed/rand_seed={}/split/train.csv'.format(args.rand_seed), index_col=0)
    ann_valid = pd.read_csv('./dataset/Fitzpatrick-17k/processed/rand_seed={}/split/val.csv'.format(args.rand_seed), index_col=0)
    ann_train.reset_index(inplace=True)
    ann_valid.reset_index(inplace=True)
    
    train_pkl = './dataset/Fitzpatrick-17k/processed/rand_seed={}/pkls/train_images.pkl'.format(args.rand_seed)
    valid_pkl = './dataset/Fitzpatrick-17k/processed/rand_seed={}/pkls/val_images.pkl'.format(args.rand_seed)


    # standard transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # augmentation transform
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128, scale=(0.4, 1),
                                     ratio=(3 / 4, 4 / 3)),
        transforms.ToTensor(),
    ])
    
    # create train dataset
    train_set = Fitz17k(dataframe=ann_train, path_to_pickles=train_pkl, sens_name='skintone', sens_classes=2, transform=transform_train)
    weights = train_set.get_weights(resample_which='group')
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = data.DataLoader(train_set,
                                    batch_size=args.batch_size,
                                    sampler=sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=True)
    # create validation dataset
    valid_set = Fitz17k(dataframe=ann_valid, path_to_pickles=valid_pkl, sens_name='skintone', sens_classes=2, transform=transform)
    valid_loader = data.DataLoader(valid_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=True)

    # GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    criterion = nn.CrossEntropyLoss()
    net = resnet152(num_classes=9, select_pos=False)
    load_weights(net)
    net = net.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.max_epoch,
                                                     eta_min=0)
    # train
    t_acc, t_loss, v_acc, v_loss = train(net, criterion, train_loader, valid_loader, optimizer, args.max_epoch, valid_interval=5)
    save_final_model(net, args.exp_id, args.rand_seed)
    save_logs(t_acc, t_loss, v_acc, v_loss, args.exp_id, args.rand_seed)
    
    

    