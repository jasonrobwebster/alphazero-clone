import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.nn as nn

from nnet.pytorch import AlphaZeroNet

mp = mp.get_context('spawn')

print("Cuda: " + str(torch.cuda.is_available()))

net = AlphaZeroNet(game, blocks=20, save_path='./models/pytorch/chess.model')
net.build()

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
