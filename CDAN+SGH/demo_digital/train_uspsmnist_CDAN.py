import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import network
import sys

def getgrad(net):
    x=[]
    for name,param in net.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                x.append(torch.tensor(param.grad).view(-1).contiguous())
    x=torch.cat(x,dim=0).detach()
    return x
def __init__(self,x,opt_x):
    self.x = x.register_hook
'''
def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
    if which_opt == 'momentum':
        self.opt_x = optim.SGD(self.x.parameters(),
                               lr=lr, weight_decay=0.0005,
                               momentum=momentum)
'''
def train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, method):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        feature, output = model(torch.cat((data_source, data_target), 0))
        #optimizer1 = optim.SGD(Variable(feature).parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
        loss1 = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(output)
        if epoch > start_epoch:
            #if method == 'CDAN-E':
              #  entropy = loss_func.Entropy(softmax_output)
             #   loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(num_iter*(epoch-start_epoch)+batch_idx), random_layer)
            #elif method == 'CDAN':
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            grad1 = getgrad(model)
            optimizer.zero_grad()
            loss_func.CDAN([feature, softmax_output], ad_net, None, None, random_layer).backward(retain_graph=True)
            grad2 = getgrad(model)
            optimizer.zero_grad()
            #print("grad1:",grad1)
            if torch.dot(grad1, grad2) < 0:
                #GA = (torch.dot(torch.tensor(grad1), torch.tensor(grad2))) / (torch.dot(torch.tensor(grad1), torch.tensor(grad1)))
                GA = (torch.dot(grad1, grad2)) / (torch.dot(grad1, grad1))
                GB = (torch.dot(grad1, grad2)) / (torch.dot(grad2, grad2))
                #GB = (torch.dot(torch.tensor(grad1), torch.tensor(grad2))) / (torch.dot(torch.tensor(grad2), torch.tensor(grad2)))
                #print('np.size(GA)')
                #print('np.size(GB)')
            else:
                GA=0
                GB=0
                #loss += loss_func.CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            hook1= model.register_backward_hook(model.aa)
            loss1.backward(retain_graph=True)
            hook1.remove()
            hook2 = model.register_backward_hook(model.aa)
            loss_func.CDAN([feature, softmax_output], ad_net, None, None, random_layer).backward(retain_graph=True)
            hook2.remove()
            loss=loss1-GA * loss1 +(1-GB) *loss_func.CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            #elif method == 'DANN':
                #loss += loss_func.DANN(feature, ad_net)
           # else:
               # raise ValueError('Method cannot be recognized.')
            loss.backward()
            optimizer.step()
        #self.opt_x.step()
        

        if epoch > start_epoch:
            optimizer_ad.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss1.item()))
def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
    parser.add_argument('method', type=str, default='CDAN', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--task', default='USPS2MNIST', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str,
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
                   
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    if args.task == 'USPS2MNIST':
        source_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/usps_train.txt'
        target_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/mnist_train.txt'
        test_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/mnist_test.txt'
        start_epoch = 1
        decay_epoch = 6
    elif args.task == 'MNIST2USPS':
        source_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/mnist_train.txt'
        target_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/usps_train.txt'
        test_list = '/media/zrway/8T/HJK/CDAN-master/data/usps2mnist/usps_test.txt'
        start_epoch = 1
        decay_epoch = 5
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    model = network.LeNet()
    model = model.cuda()
    class_num = 10

    if args.random:
        random_layer = network.RandomLayer([model.output_num(), class_num], 500)
        ad_net = network.AdversarialNetwork(500, 500)
        random_layer.cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num() * class_num, 500)
    ad_net = ad_net.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    for epoch in range(1, args.epochs + 1):
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, args.method)
        test(args, model, test_loader)
        
        state = {"model": model, "ad_net": ad_net,"optimizer": optimizer,"optimizer_ad":optimizer_ad,"random_layer":random_layer,"epoch":epoch}
        torch.save(obj=state, f='/media/zrway/8T/HJK/CDAN-master/save_model/'+str(epoch)+'train_uspsmnist_CDAN.pth')
if __name__ == '__main__':
    main()
