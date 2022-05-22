
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
from torchvision.datasets import CIFAR10,MNIST,SVHN

from gradalign import get_input_grad, l2_norm_batch
from custom_op_train import * 
from trades_loss import trades_loss

parser = argparse.ArgumentParser(description='PyTorch Adversarail Training')
##################################### Dataset #################################################
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--arch', type=str, default=None, help='model Architecture')
parser.add_argument('--weight_dir', type=str, default=None, help='weight direction')
parser.add_argument('--mask_dir', type=str, default=None, help='mask direction')
parser.add_argument('--normalize_v2', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--a_init', default=0.4, type=float, help='initial a for grafting')
parser.add_argument('--b_init', default=0, type=float, help='initial b for grafting')
##################################### General setting ############################################
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lr_weight', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
##################################### Training Mode ###########################################
parser.add_argument('--train_norm', default='l_inf', type=str, help='l_inf or l_2')
parser.add_argument('--train_eps', default=(2/255), type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=1, type=int, help='itertion number of attack during training')
parser.add_argument('--train_alpha', default=(2.5/255), type=float, help='step size of attack during training')
parser.add_argument('--test_norm', default='l_inf', type=str, help='l_inf or l_2')
parser.add_argument('--test_eps', default=(2/255), type=float, help='epsilon of attack during training')
parser.add_argument('--test_step', default=1, type=int, help='itertion number of attack during training')
parser.add_argument('--test_alpha', default=(2.5/255), type=float, help='step size of attack during training')
parser.add_argument('--grad_align_cos_lambda', default=0.2, type=float, help='hyperparameters for GradAlign')
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--trades', action='store_true')

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

    model = eval(args.arch)(v=args.a_init, b=args.b_init)

    if args.weight_dir:
        checkpoint = torch.load(args.weight_dir, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        new_dict = {}
        for key in checkpoint.keys():
            if 'resnet4b' in args.arch:
                new_key = key 
            else:
                new_key = 'layer{}'.format(key)
            assert new_key in model.state_dict().keys()
            new_dict[new_key] = checkpoint[key]
        model.load_state_dict(new_dict, strict=False)
        print('Loading Pretrained Weight from {}'.format(args.weight_dir))

    if args.mask_dir:
        mask_file = torch.load(args.mask_dir, map_location='cpu')
        idx = 0
        mask_checkpoint = {}
        for key in model.state_dict().keys():
            if '.mask' in key:
                mask_checkpoint[key] = mask_file[idx]
                idx += 1 
        model.load_state_dict(mask_checkpoint, strict=False)
        print('Loading Mask from {}'.format(args.mask_dir))

    model = nn.Sequential(normalize, model)
    model.cuda()

    print(model)
    check_sparsity_model(model.state_dict())

    for name, p in model.named_parameters():
        if '.linear_masked_relu' in name:
            print('A', name)

    optimizer_grouped_parameters = [
        {'params': [v for k, v in model.named_parameters() if '.linear_masked_relu' in k], 'weight_decay': 0, 'lr': args.lr},
        {'params': [v for k, v in model.named_parameters() if not '.linear_masked_relu' in k], 'lr': args.lr_weight, 'weight_decay': args.weight_decay},
    ]
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()


    if args.eval:
        # Test against PGD-100
        args.test_step = 100
        args.test_alpha = 2.5 * args.test_eps / args.test_step    
        args.test_alpha = args.test_eps / 4
        print('Test with eps = {}, step = {}, alpha = {}'.format(args.test_eps, args.test_step, args.test_alpha))    

        model_weight = torch.load(os.path.join(args.save_dir, 'model_RA_best.pth.tar'))['state_dict']
        model[1].load_state_dict(model_weight)
        test_tacc = test(test_loader, model, criterion, args)
        test_racc = test_adv(test_loader, model, criterion, args)
        print('* RA best \t SA = {:.4f} \t PGD-100 = {:.4f}'.format(test_tacc, test_racc))

        return 


    all_result = {}
    all_result['train_acc'] = []
    all_result['test_ta'] = []
    all_result['test_ra'] = []
    all_result['val_ta'] = []
    all_result['val_ra'] = []
    best_sa = 0
    best_ra = 0
    start_epoch = 0


    for epoch in range(start_epoch, args.epochs):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args)
        all_result['train_acc'].append(acc)
        scheduler.step()

        # Evaluation
        test_tacc = test(test_loader, model, criterion, args)
        all_result['test_ta'].append(test_tacc)
        val_tacc = test(val_loader, model, criterion, args)
        all_result['val_ta'].append(val_tacc)
        is_sa_best = val_tacc > best_sa
        best_sa = max(val_tacc, best_sa)

        test_racc = test_adv(test_loader, model, criterion, args)
        all_result['test_ra'].append(test_racc)
        val_racc = test_adv(val_loader, model, criterion, args)
        all_result['val_ra'].append(val_racc)
        is_ra_best = val_racc > best_ra 
        best_ra = max(val_racc, best_ra)

        best_name_list = []
        if is_sa_best: best_name_list.append('model_SA_best.pth.tar')
        if is_ra_best: best_name_list.append('model_RA_best.pth.tar')

        checkpoint_state = {
            'best_sa': best_sa,
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model[1].state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'result': all_result
        }
        save_checkpoint(checkpoint_state, args.save_dir, best_name=best_name_list)

    best_ta_epoch = np.argmax(np.array(all_result['val_ta']))
    best_ra_epoch = np.argmax(np.array(all_result['val_ra']))
    print('* best TA model // SA = {:.4f}, RA = {:.4f}, Epoch = {}'.format(all_result['test_ta'][best_ta_epoch], all_result['test_ra'][best_ta_epoch], best_ta_epoch+1))
    print('* best RA model // SA = {:.4f}, RA = {:.4f}, Epoch = {}'.format(all_result['test_ta'][best_ra_epoch], all_result['test_ra'][best_ra_epoch], best_ra_epoch+1))

    # Test against PGD-100
    args.test_step = 100
    args.test_alpha = 2.5 * args.test_eps / args.test_step    
    print('Test with eps = {}, step = {}, alpha = {}'.format(args.test_eps, args.test_step, args.test_alpha))

    model_weight = torch.load(os.path.join(args.save_dir, 'model_RA_best.pth.tar'))['state_dict']
    model[1].load_state_dict(model_weight)
    test_tacc = test(test_loader, model, criterion, args)
    test_racc = test_adv(test_loader, model, criterion, args)
    print('* RA best \t SA = {:.4f} \t PGD-100 = {:.4f}'.format(test_tacc, test_racc))


def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()

    if args.train_eps != 0.0:
        print('Adversarial training')
    else:
        print('Standard Training')

    if args.grad_align_cos_lambda != 0.0:
        print('Using GradAlign')

    if args.trades:
        print('ADV Training with Trades')

    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()
        if args.trades:
            loss, output_adv = trades_loss(model, image, target, optimizer, step_size=args.train_alpha,
                epsilon=args.train_eps, perturb_steps=args.train_step, beta=args.beta, distance=args.train_norm)
        else:
            if args.train_eps != 0.0:
                #adv samples
                model.eval() # https://arxiv.org/pdf/2010.00467.pdf
                delta = attack_pgd(model, image, target, args.train_eps, args.train_alpha, args.train_step, args.train_norm)
                delta.detach()
                adv_delta = torch.clamp(image + delta, 0, 1) - image
                model.train()
                # compute output
                output_adv = model(image + adv_delta)
            else:
                output_adv = model(image) # standard training

            loss = criterion(output_adv, target)
            
            # GradAlign Regularization
            if args.grad_align_cos_lambda != 0.0:
                grad = get_input_grad(model, image, target, eps=args.train_eps, delta_init='none', backprop=False)
                grad2 = get_input_grad(model, image, target, eps=args.train_eps, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg = args.grad_align_cos_lambda * (1.0 - cos.mean())
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

    print('adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


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


def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def check_sparsity_model(model_weight):
    remain_ele = 0
    all_ele = 0
    for key in model_weight.keys():
        if '.mask' in key:
            print(key)
            remain_ele += model_weight[key].sum().item()
            all_ele += model_weight[key].nelement()
    print('Grafting Ratio = {:.2f}%'.format(100 * remain_ele / all_ele))


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


