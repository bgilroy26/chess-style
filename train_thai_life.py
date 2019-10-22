import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from thai_life import ThaiLife
import sys

parser = argparse.ArgumentParser(description='Mae Toi')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=.01, metavar='N',
                    help='learning rate (default: .01)')
parser.add_argument('--decay', type=int, default=.99, metavar='N',
                    help='decay rate of learning rate (default: .99)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

lr = args.lr
decay = args.decay
batch_size = args.batch_size

print('lr: {} | decay: {} | batch size: {}'.format(lr, decay, batch_size))

print('Loading data...')
positions1 = np.load('../chessfiles/train.npy')
positions2 = np.load('../chessfiles/test.npy')
positions3 = np.vstack((positions1, positions2))
results = np.load('../chessfiles/labels.npy')

p = np.random.permutation(results.shape[0])
positions = positions3[p]
results = results[p]

four_fifths = int(positions.shape[0]*.8)
train_positions = positions[:four_fifths]
train_results = results[:four_fifths]
test_positions = positions[four_fifths:]
test_results = results[four_fifths:]

class TrainSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        rnd_idx = np.random.randint(0, train_positions.shape[0])
        rand_position = train_positions[rnd_idx]
        rand_label = np.array(np.argmax(train_results[rnd_idx]))

        stacked = torch.from_numpy(rand_position).type(torch.FloatTensor)
        label = torch.from_numpy(rand_label).type(torch.LongTensor)
        return (stacked, label)

    def __len__(self):
        return train_positions.shape[0]

class TestSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        rnd_idx = np.random.randint(0, test_positions.shape[0])
        rand_position = test_positions[rnd_idx]
        rand_label = np.array(np.argmax(test_results[rnd_idx]))

        stacked = torch.from_numpy(rand_position).type(torch.FloatTensor)
        label = torch.from_numpy(rand_label).type(torch.LongTensor)
        return (stacked, label)


    def __len__(self):
        return test_positions.shape[0]

train_loader = torch.utils.data.DataLoader(TrainSet(),batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(),batch_size=batch_size, shuffle=True)


print('Buidling model...')
model = ThaiLife().to(device)
for idx, child in enumerate(model.children()):
    if idx not in (3, 4, 5, 6):
        for param in child.parameters():
            param.requires_grad = False


optimizer = optim.Adam(model.parameters(), lr=lr)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(pred, label):
    CE = F.cross_entropy(pred, label, reduction='mean')
    return CE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, label)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def get_acc():
    e = enumerate(test_loader)
    correct = 0
    for batch_idx, (data, label) in e:
        pred = model(data.to(device))
        correct += np.sum((pred > .5).cpu().detach().numpy() * label.numpy())
    return correct / float(test_loader.dataset.length)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            test_loss += loss_function(pred, label).mean().item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = 'checkpoints/thai_life/lr_{}_decay_{}'.format(int(lr*1000), int(decay*100))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, 'evaluator_{}_epoch.pth.tar'.format(epoch)))
    torch.save(state, './checkpoints/best_evaluator.pth.tar')

start_epoch = 1
resume = True
if resume:
    state = torch.load('./checkpoints/best_evaluator.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    for idx, child in enumerate(model.children()):
        if idx not in (3, 4, 5, 6):
            for param in child.parameters():
                param.requires_grad = False

    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']

print('Begin train...')
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    test(epoch)
    save(epoch)

    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay

