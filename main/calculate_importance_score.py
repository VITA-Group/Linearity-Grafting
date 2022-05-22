
import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, SVHN


from gradalign import get_input_grad, l2_norm_batch
from trades_loss import trades_loss
from utils_importance_score import * 


parser = argparse.ArgumentParser(description='Calculate Pruning Scores')
##################################### Dataset #################################################
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--arch', type=str, default=None, help='model Architecture')
parser.add_argument('--mode', type=str, default='magnitude', help='Types for generate activation masks')
parser.add_argument('--loss', type=str, default='std')
parser.add_argument('--remain_ratio', default=0.5, type=float, help='Remaining ratio of neurons')
parser.add_argument('--global_prune', action='store_true', help='Global Pruning')
parser.add_argument('--weight_dir', type=str, default=None, help='Mask direction')
parser.add_argument('--lr_score', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--epochs_score', default=5, type=int, help='number of total epochs to run')
##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
##################################### Training Mode ###########################################
parser.add_argument('--test_norm', default='l_inf', type=str, help='l_inf or l_2')
parser.add_argument('--test_eps', default=(2/255), type=float, help='epsilon of attack during training')
parser.add_argument('--test_step', default=1, type=int, help='itertion number of attack during training')
parser.add_argument('--test_alpha', default=(2.5/255), type=float, help='step size of attack during training')
parser.add_argument('--normalize_v2', action='store_true')


def main():
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    if args.dataset == 'cifar10':

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(45000)))
        val_set = Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
        test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        if args.normalize_v2:
            MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
            STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
        else:
            MEANS = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        normalize = NormalizeByChannelMeanStd(mean=MEANS, std=STD)

    elif args.dataset == 'mnist':

        train_set = Subset(MNIST(args.data, train=True, transform=transforms.ToTensor(), download=True), list(range(54000)))
        val_set = Subset(MNIST(args.data, train=True, transform=transforms.ToTensor(), download=True), list(range(54000, 60000)))
        test_set = MNIST(args.data, train=False, transform=transforms.ToTensor(), download=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        MEANS = np.array([0], dtype=np.float32)
        STD = np.array([1], dtype=np.float32)
        normalize = NormalizeByChannelMeanStd(mean=MEANS, std=STD)

    elif args.dataset == 'svhn':

        train_set = Subset(SVHN(args.data, split='train', transform=transforms.ToTensor(), download=True),list(range(68257)))
        val_set = Subset(SVHN(args.data, split='train', transform=transforms.ToTensor(), download=True),list(range(68257,73257)))
        test_set = SVHN(args.data, split='test', transform=transforms.ToTensor(), download=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        MEANS = np.array([0.4377, 0.4438, 0.4728], dtype=np.float32)
        STD = np.array([0.1201, 0.1231, 0.1052], dtype=np.float32)
        normalize = NormalizeByChannelMeanStd(mean=MEANS, std=STD)

    else:
        raise ValueError('unsupport datast')

    model_hook = eval(args.arch)()

    # load weight 
    if args.weight_dir:
        print('Loading weight from {}'.format(args.weight_dir))
        pretrained_weight = torch.load(args.weight_dir, map_location='cpu')
        if 'state_dict' in pretrained_weight.keys():
            pretrained_weight = pretrained_weight['state_dict']

        new_dict = {}
        for key in pretrained_weight.keys():
            if 'resnet4b' in args.arch:
                new_key = key 
            else:
                new_key = 'layer{}'.format(key)
            print(new_key)
            assert new_key in model_hook.state_dict().keys()
            new_dict[new_key] = pretrained_weight[key]
        model_hook.load_state_dict(new_dict, strict=False)

    model_hook = nn.Sequential(normalize, model_hook)
    model_hook.cuda()
    print(model_hook)

    criterion = nn.CrossEntropyLoss()

    # Double Check: evaludate accuracy
    ACC = test(test_loader, model_hook, criterion, args)
    print('* Model Hook Accuracy = {:.2f}'.format(ACC))

    if args.mode == 'gradient':
        mask_score = calculate_activation_mask_act_gradient(model_hook, args, Hard_mask=False)
        torch.save(mask_score, os.path.join(args.save_dir,'score_gradient_{}_{}_new.pth'.format(args.arch, args.loss)))
    elif args.mode == 'magnitude':
        mask_score = calculate_activation_mask_act_magnitude(model_hook, args, Hard_mask=False)
        torch.save(mask_score, os.path.join(args.save_dir,'score_magnitude_{}_{}_new.pth'.format(args.arch, args.loss)))


def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        delta = attack_pgd(model, image, target, args.test_eps, args.test_alpha, args.test_step, args.test_norm)
        delta.detach()
        image_adv = torch.clamp(image + delta[:image.size(0)], 0, 1)

        # compute output
        with torch.no_grad():
            output = model(image_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm="l_inf", 
                early_stop=False, restarts=1, randominit=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if randominit:
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta) # add the normalize operation inside model
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def train_epoch_prune_fgsm_gradalign(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()

    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        model.eval() # https://arxiv.org/pdf/2010.00467.pdf
        delta = attack_pgd(model, image, target, 2/255, 2.5/255, 1, 'l_inf')
        delta.detach()
        adv_delta = torch.clamp(image + delta, 0, 1) - image
        model.train()
        # compute output
        output_adv = model(image + adv_delta)

        loss = criterion(output_adv, target)
        
        # GradAlign Regularization
        grad = get_input_grad(model, image, target, eps=2/255, delta_init='none', backprop=False)
        grad2 = get_input_grad(model, image, target, eps=2/255, delta_init='random_uniform', backprop=True)
        grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
        grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
        grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
        reg = 0.2 * (1.0 - cos.mean())
        loss += reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('pruning adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def train_epoch_prune_pgd(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()

    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        model.eval() # https://arxiv.org/pdf/2010.00467.pdf
        delta = attack_pgd(model, image, target, 2/255, 5/(7*255), 7, 'l_inf')
        delta.detach()
        adv_delta = torch.clamp(image + delta, 0, 1) - image
        model.train()
        # compute output
        output_adv = model(image + adv_delta)
        loss = criterion(output_adv, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('pruning adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def train_epoch_prune_trades(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()

    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()
        loss, output_adv = trades_loss(model, image, target, optimizer, step_size=5/(7*255), epsilon=2/255, perturb_steps=7, beta=6.0, distance='l_inf')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('pruning adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar', best_name=None):
    
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if best_name:
        for keyname in best_name:
            shutil.copyfile(filepath, os.path.join(save_path, keyname))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()


