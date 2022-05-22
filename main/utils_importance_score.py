import os 
import torch 
import torch.nn as nn 
import numpy as np 
from torch.nn import functional as F
from torchvision.datasets import CIFAR10, MNIST, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from calculate_importance_score import test, test_adv

__all__ = ['calculate_activation_mask_act_gradient', 'calculate_activation_mask_act_magnitude',
        'mnist_8_200_hook', 'mnist_8_200_hook_pre', 'mnist_conv_big_hook', 'mnist_conv_big_hook_pre',
        'svhn_conv_big_hook', 'svhn_conv_big_hook_pre',
        'cifar_cnn_b_hook', 'cifar_cnn_b_hook_pre', 'cifar_cnn_17m_hook', 'cifar_cnn_17m_hook_pre',
        'cifar_conv_big_hook', 'cifar_conv_big_hook_pre', 'cifar_resnet4b_hook', 'cifar_resnet4b_hook_pre']


def calculate_activation_mask_act_gradient(model_hook, args, Hard_mask=False):
    np.random.seed(1)
    if args.dataset == 'cifar10':
        index = np.random.permutation(50000)[:1000]
        calculate_set = Subset(CIFAR10(args.data, train=True, transform=transforms.ToTensor(), download=True),list(index))
    elif args.dataset == 'mnist':
        index = np.random.permutation(60000)[:1000]
        calculate_set = Subset(MNIST(args.data, train=True, transform=transforms.ToTensor(), download=True),list(index))
    elif args.dataset == 'svhn':
        index = np.random.permutation(73257)[:1000]
        calculate_set = Subset(SVHN(args.data, split='train', transform=transforms.ToTensor(), download=True),list(index))
    else:
        raise ValueError('Not support')
    calculate_loader = DataLoader(calculate_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    all_grad = []
    model_hook.eval() 
    model_hook[1].store = True

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for batch_idx, (data, target) in enumerate(calculate_loader):
        data = data.cuda()
        target = target.cuda()
        if args.loss == 'std':
            output = model_hook(data)
            loss = criterion(output, target)
        elif args.loss == 'fgsm':
            model_hook[1].store = False 
            delta = attack_pgd(model_hook, data, target, args.test_eps, args.test_alpha, args.test_step, args.test_norm)
            delta.detach()
            data_adv = torch.clamp(data + delta[:data.size(0)], 0, 1)
            model_hook[1].store = True
            output = model_hook(data_adv)
            loss = criterion(output, target)

        model_hook.zero_grad()
        loss.backward()

        all_grad.append(model_hook[1].gradient_list)
    model_hook[1].store = False

    if Hard_mask:
        print('Apply Hard Mask')
        mask_list = []
        for key in model_hook.state_dict().keys():
            if '.mask' in key:
                print('mask = {}'.format(key))
                mask_list.append(model_hook.state_dict()[key])

    # calculate scores
    batch_num = len(all_grad)
    layer_num = len(all_grad[0])
    new_scores = []
    for idx in range(layer_num):
        layer_grad = []
        for batch in range(batch_num):
            layer_grad.append(all_grad[batch][-(idx+1)].detach())
        layer_grad = torch.cat(layer_grad, dim=0)
        
        if Hard_mask:
            layer_grad = layer_grad * mask_list[idx]

        # abs of grad * activation
        score = layer_grad.abs().mean(0)
        new_scores.append(score)

    # generate mask 
    if args.global_prune:
        print('Generate Global Mask')
        output_mask = global_mask(new_scores, args.remain_ratio)
    else:        
        print('Generate Layer-wise Mask')
        output_mask = local_mask(new_scores, args.remain_ratio)
    # check_sparsity(output_mask)

    return new_scores 

def calculate_activation_mask_act_magnitude(model_hook, args, Hard_mask=False):
    np.random.seed(1)
    if args.dataset == 'cifar10':
        index = np.random.permutation(50000)[:1000]
        calculate_set = Subset(CIFAR10(args.data, train=True, transform=transforms.ToTensor(), download=True),list(index))
    elif args.dataset == 'mnist':
        index = np.random.permutation(60000)[:1000]
        calculate_set = Subset(MNIST(args.data, train=True, transform=transforms.ToTensor(), download=True),list(index))
    elif args.dataset == 'svhn':
        index = np.random.permutation(73257)[:1000]
        calculate_set = Subset(SVHN(args.data, split='train', transform=transforms.ToTensor(), download=True),list(index))
    else:
        raise ValueError('Not support')
    calculate_loader = DataLoader(calculate_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    all_grad = []
    all_activation = []
    model_hook.eval() 
    model_hook[1].store = True

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for batch_idx, (data, target) in enumerate(calculate_loader):
        data = data.cuda()
        target = target.cuda()
        if args.loss == 'std':
            output = model_hook(data)
            loss = criterion(output, target)
        elif args.loss == 'fgsm':
            model_hook[1].store = False 
            delta = attack_pgd(model_hook, data, target, args.test_eps, args.test_alpha, args.test_step, args.test_norm)
            delta.detach()
            data_adv = torch.clamp(data + delta[:data.size(0)], 0, 1)
            model_hook[1].store = True
            output = model_hook(data_adv)
            loss = criterion(output, target)

        model_hook.zero_grad()
        loss.backward()
        all_grad.append(model_hook[1].gradient_list)
        all_activation.append(model_hook[1].activation_list)

    model_hook[1].store = False

    if Hard_mask:
        print('Apply Hard Mask')
        mask_list = []
        for key in model_hook.state_dict().keys():
            if '.mask' in key:
                print('mask = {}'.format(key))
                mask_list.append(model_hook.state_dict()[key])

    # calculate scores
    batch_num = len(all_grad)
    layer_num = len(all_grad[0])
    new_scores = []
    for idx in range(layer_num):
        layer_act = []
        for batch in range(batch_num):
            layer_act.append(all_activation[batch][idx].detach())
        layer_act = torch.cat(layer_act, dim=0)
        
        if Hard_mask:
            layer_act = layer_act * mask_list[idx]

        # abs of grad * activation
        score = layer_act.abs().mean(0)
        new_scores.append(score)

    # generate mask 
    if args.global_prune:
        print('Generate Global Mask')
        output_mask = global_mask(new_scores, args.remain_ratio)
    else:        
        print('Generate Layer-wise Mask')
        output_mask = local_mask(new_scores, args.remain_ratio)
    # check_sparsity(output_mask)

    return new_scores 


# Models 
class Relu_Mask(nn.Module):

    def __init__(self, size):
        super(Relu_Mask, self).__init__()
        self.register_buffer('mask', torch.ones(size))

    def forward(self, x):
        return F.relu(x) * self.mask 

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class mnist_8_200_hook(nn.Module):

    def __init__(self):
        super(mnist_8_200_hook, self).__init__()

        self.layer0 = Flatten()
        self.layer1 = nn.Linear(784, 200)
        self.layer2 = Relu_Mask(size=(200))
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = Relu_Mask(size=(200))
        self.layer5 = nn.Linear(200, 200)
        self.layer6 = Relu_Mask(size=(200))
        self.layer7 = nn.Linear(200, 200)
        self.layer8 = Relu_Mask(size=(200))
        self.layer9 = nn.Linear(200, 200)
        self.layer10 = Relu_Mask(size=(200))
        self.layer11 = nn.Linear(200, 200)
        self.layer12 = Relu_Mask(size=(200))
        self.layer13 = nn.Linear(200, 200)
        self.layer14 = Relu_Mask(size=(200))
        self.layer15 = nn.Linear(200, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer3(out)
        out = self.layer4(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer5(out)
        out = self.layer6(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer7(out)
        out = self.layer8(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer9(out)
        out = self.layer10(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer11(out)
        out = self.layer12(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer13(out)
        out = self.layer14(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer15(out)
        return out

class mnist_8_200_hook_pre(nn.Module):

    def __init__(self):
        super(mnist_8_200_hook_pre, self).__init__()

        self.layer0 = Flatten()
        self.layer1 = nn.Linear(784, 200)
        self.layer2 = Relu_Mask(size=(200))
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = Relu_Mask(size=(200))
        self.layer5 = nn.Linear(200, 200)
        self.layer6 = Relu_Mask(size=(200))
        self.layer7 = nn.Linear(200, 200)
        self.layer8 = Relu_Mask(size=(200))
        self.layer9 = nn.Linear(200, 200)
        self.layer10 = Relu_Mask(size=(200))
        self.layer11 = nn.Linear(200, 200)
        self.layer12 = Relu_Mask(size=(200))
        self.layer13 = nn.Linear(200, 200)
        self.layer14 = Relu_Mask(size=(200))
        self.layer15 = nn.Linear(200, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer4(out)
        out = self.layer5(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer6(out)
        out = self.layer7(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer8(out)
        out = self.layer9(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer10(out)
        out = self.layer11(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer12(out)
        out = self.layer13(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer14(out)
        out = self.layer15(out)
        return out


class mnist_conv_big_hook(nn.Module):

    def __init__(self):
        super(mnist_conv_big_hook, self).__init__()

        self.layer0 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,28,28))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,14,14))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,14,14))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,7,7))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(3136, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer2(out)
        out = self.layer3(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer4(out)
        out = self.layer5(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer6(out)
        out = self.layer7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer11(out)
        out = self.layer12(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer13(out)
        return out

class mnist_conv_big_hook_pre(nn.Module):

    def __init__(self):
        super(mnist_conv_big_hook_pre, self).__init__()

        self.layer0 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,28,28))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,14,14))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,14,14))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,7,7))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(3136, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer1(out)
        out = self.layer2(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer10(out)
        out = self.layer11(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out


class svhn_conv_big_hook(nn.Module):

    def __init__(self):
        super(svhn_conv_big_hook, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,32,32))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,16,16))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,16,16))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,8,8))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer2(out)
        out = self.layer3(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer4(out)
        out = self.layer5(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer6(out)
        out = self.layer7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer11(out)
        out = self.layer12(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer13(out)
        return out

class svhn_conv_big_hook_pre(nn.Module):

    def __init__(self):
        super(svhn_conv_big_hook_pre, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,32,32))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,16,16))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,16,16))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,8,8))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer1(out)
        out = self.layer2(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer10(out)
        out = self.layer11(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out


class cifar_cnn_b_hook(nn.Module):

    def __init__(self):
        super(cifar_cnn_b_hook, self).__init__()

        self.layer0 = nn.ZeroPad2d((1,2,1,2))
        self.layer1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=0)
        self.layer2 = Relu_Mask(size=(32,16,16))
        self.layer3 = nn.Conv2d(32, 128, (4,4), stride=2, padding=1)
        self.layer4 = Relu_Mask(size=(128,8,8))
        self.layer5 = Flatten()
        self.layer6 = nn.Linear(8192, 250)
        self.layer7 = Relu_Mask(size=(250))
        self.layer8 = nn.Linear(250, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer3(out)
        out = self.layer4(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer8(out)
        return out

class cifar_cnn_b_hook_pre(nn.Module):

    def __init__(self):
        super(cifar_cnn_b_hook_pre, self).__init__()

        self.layer0 = nn.ZeroPad2d((1,2,1,2))
        self.layer1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=0)
        self.layer2 = Relu_Mask(size=(32,16,16))
        self.layer3 = nn.Conv2d(32, 128, (4,4), stride=2, padding=1)
        self.layer4 = Relu_Mask(size=(128,8,8))
        self.layer5 = Flatten()
        self.layer6 = nn.Linear(8192, 250)
        self.layer7 = Relu_Mask(size=(250))
        self.layer8 = nn.Linear(250, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out


class cifar_cnn_17m_hook(nn.Module):

    def __init__(self):
        super(cifar_cnn_17m_hook, self).__init__()

        self.layer0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(64,32,32))
        self.layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.layer3 = Relu_Mask(size=(64,32,32))
        self.layer4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer5 = Relu_Mask(size=(128,16,16))
        self.layer6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer7 = Relu_Mask(size=(128,16,16))
        self.layer8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer9 = Relu_Mask(size=(128,16,16))
        self.layer10 = Flatten()
        self.layer11 = nn.Linear(32768, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer2(out)
        out = self.layer3(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer4(out)
        out = self.layer5(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer6(out)
        out = self.layer7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer8(out)
        out = self.layer9(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer13(out)
        return out

class cifar_cnn_17m_hook_pre(nn.Module):

    def __init__(self):
        super(cifar_cnn_17m_hook_pre, self).__init__()

        self.layer0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(64,32,32))
        self.layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.layer3 = Relu_Mask(size=(64,32,32))
        self.layer4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer5 = Relu_Mask(size=(128,16,16))
        self.layer6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer7 = Relu_Mask(size=(128,16,16))
        self.layer8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer9 = Relu_Mask(size=(128,16,16))
        self.layer10 = Flatten()
        self.layer11 = nn.Linear(32768, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer1(out)
        out = self.layer2(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer7(out)
        out = self.layer8(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out


class cifar_conv_big_hook(nn.Module):

    def __init__(self):
        super(cifar_conv_big_hook, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,32,32))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,16,16))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,16,16))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,8,8))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)
        out = self.layer1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer2(out)
        out = self.layer3(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer4(out)
        out = self.layer5(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer6(out)
        out = self.layer7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer11(out)
        out = self.layer12(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.layer13(out)
        return out

class cifar_conv_big_hook_pre(nn.Module):

    def __init__(self):
        super(cifar_conv_big_hook_pre, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.layer1 = Relu_Mask(size=(32,32,32))
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.layer3 = Relu_Mask(size=(32,16,16))
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.layer5 = Relu_Mask(size=(64,16,16))
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.layer7 = Relu_Mask(size=(64,8,8))
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.layer10 = Relu_Mask(size=(512))
        self.layer11 = nn.Linear(512, 512)
        self.layer12 = Relu_Mask(size=(512))
        self.layer13 = nn.Linear(512, 10)

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.layer0(x)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer1(out)
        out = self.layer2(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer10(out)
        out = self.layer11(out)

        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out



class cifar_resnet4b_hook_layer1_0(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer1_0, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=2)
        )
    def forward(self, x):
        return x

class cifar_resnet4b_hook_layer1_1(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer1_1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

class cifar_resnet4b_hook_layer2_0(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer2_0, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 32, 1, stride=2)
        )

class cifar_resnet4b_hook_layer2_1(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer2_1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

class cifar_resnet4b_hook(nn.Module):

    def __init__(self):
        super(cifar_resnet4b_hook, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            cifar_resnet4b_hook_layer1_0(),
            cifar_resnet4b_hook_layer1_1())
        self.layer2 = nn.Sequential(
            cifar_resnet4b_hook_layer2_0(),
            cifar_resnet4b_hook_layer2_1())
        self.linear1 = nn.Linear(512, 100)
        self.linear2 = nn.Linear(100, 10)

        self.relu1 = Relu_Mask(size=(16,16,16))
        self.relu2 = Relu_Mask(size=(32,8,8))
        self.relu3 = Relu_Mask(size=(32,8,8))
        self.relu4 = Relu_Mask(size=(32,8,8))
        self.relu5 = Relu_Mask(size=(32,8,8))
        self.relu6 = Relu_Mask(size=(32,4,4))
        self.relu7 = Relu_Mask(size=(32,4,4))
        self.relu8 = Relu_Mask(size=(32,4,4))
        self.relu9 = Relu_Mask(size=(32,4,4))
        self.relu10 = Relu_Mask(size=(100))

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.conv1(x)
        out = self.relu1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        # layer1-0
        out_sub = self.layer1[0].conv1(out)
        out_sub = self.relu2(out_sub)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.layer1[0].conv2(out_sub)
        out = out_sub + self.layer1[0].shortcut(out)
        out = self.relu3(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        # layer1-1
        out_sub = self.layer1[1].conv1(out)
        out_sub = self.relu4(out_sub)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.layer1[1].conv2(out_sub)
        out = out_sub + self.layer1[1].shortcut(out)
        out = self.relu5(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        # layer2-0
        out_sub = self.layer2[0].conv1(out)
        out_sub = self.relu6(out_sub)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.layer2[0].conv2(out_sub)
        out = out_sub + self.layer2[0].shortcut(out)
        out = self.relu7(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        # layer2-1
        out_sub = self.layer2[1].conv1(out)
        out_sub = self.relu8(out_sub)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.layer2[1].conv2(out_sub)
        out = out_sub + self.layer2[1].shortcut(out)
        out = self.relu9(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = torch.flatten(out, 1)
        out = self.relu10(self.linear1(out))
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)

        out = self.linear2(out)
        return out 

class cifar_resnet4b_hook_pre(nn.Module):

    def __init__(self):
        super(cifar_resnet4b_hook_pre, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            cifar_resnet4b_hook_layer1_0(),
            cifar_resnet4b_hook_layer1_1())
        self.layer2 = nn.Sequential(
            cifar_resnet4b_hook_layer2_0(),
            cifar_resnet4b_hook_layer2_1())
        self.linear1 = nn.Linear(512, 100)
        self.linear2 = nn.Linear(100, 10)

        self.relu1 = Relu_Mask(size=(16,16,16))
        self.relu2 = Relu_Mask(size=(32,8,8))
        self.relu3 = Relu_Mask(size=(32,8,8))
        self.relu4 = Relu_Mask(size=(32,8,8))
        self.relu5 = Relu_Mask(size=(32,8,8))
        self.relu6 = Relu_Mask(size=(32,4,4))
        self.relu7 = Relu_Mask(size=(32,4,4))
        self.relu8 = Relu_Mask(size=(32,4,4))
        self.relu9 = Relu_Mask(size=(32,4,4))
        self.relu10 = Relu_Mask(size=(100))

        self.store = False
        self.gradient_list = []
        self.activation_list = []

    def _store_grad(self, grad): 
        self.gradient_list.append(grad)

    def forward(self, x):

        if self.store:
            self.gradient_list = []
            self.activation_list = []

        out = self.conv1(x)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu1(out)

        # layer1-0
        out_sub = self.layer1[0].conv1(out)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.relu2(out_sub)
        out_sub = self.layer1[0].conv2(out_sub)
        out = out_sub + self.layer1[0].shortcut(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu3(out)
        # layer1-1
        out_sub = self.layer1[1].conv1(out)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.relu4(out_sub)
        out_sub = self.layer1[1].conv2(out_sub)
        out = out_sub + self.layer1[1].shortcut(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu5(out)
        # layer2-0
        out_sub = self.layer2[0].conv1(out)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.relu6(out_sub)
        out_sub = self.layer2[0].conv2(out_sub)
        out = out_sub + self.layer2[0].shortcut(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu7(out)
        # layer2-1
        out_sub = self.layer2[1].conv1(out)
        if self.store:
            out_sub.register_hook(self._store_grad)
            self.activation_list.append(out_sub)
        out_sub = self.relu8(out_sub)    
        out_sub = self.layer2[1].conv2(out_sub)
        out = out_sub + self.layer2[1].shortcut(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu9(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        if self.store:
            out.register_hook(self._store_grad)
            self.activation_list.append(out)
        out = self.relu10(out)
        out = self.linear2(out)
        return out 



# Others 
def global_mask(score_list, remain_ratio):

    # concate
    global_scores = torch.cat([score.reshape(-1) for score in score_list])
    all_elements = global_scores.shape[0]
    remain_elements = int(all_elements * remain_ratio)
    global_mask = torch.zeros_like(global_scores)

    enable_index = global_scores.sort()[1][-remain_elements:]
    global_mask[enable_index] = 1

    # recover shape
    mask_list = []
    point = 0
    for idx in range(len(score_list)):
        layer_ele = score_list[idx].reshape(-1).shape[0]
        layer_mask = global_mask[point:point+layer_ele].reshape(score_list[idx].shape)
        point += layer_ele
        mask_list.append(layer_mask)

    return mask_list

def local_mask(score_list, remain_ratio):

    mask_list = []
    for score in score_list:
        mask = torch.zeros_like(score)
        remain_elements = int(mask.nelement() * remain_ratio)
        index = score.reshape(-1).sort()[1][-remain_elements:]
        mask.reshape(-1)[index] = 1
        mask_list.append(mask)

    return mask_list

def check_sparsity(mask_list):
    remain_ele = 0
    all_ele = 0
    for mask in mask_list:
        remain_ele += mask.sum().item()
        all_ele += mask.nelement()
    print('Remain Neuron Ratio = {:.2f}%'.format(100 * remain_ele / all_ele))


# ATTACK Method
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



