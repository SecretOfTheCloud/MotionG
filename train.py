import os
import torch
import random
import numpy as np
import torch.nn as nn
from model import Model
import torch.optim as optim
from dataset import dataloader
from normalize import normalize
import copy


def build_path(path):
    for i in path:
        if not os.path.exists(i): os.makedirs(i)

def train_prepare(batch, hidden, dropout, device, path, learn_rate, test_rate):
    print(device)
    data = np.load(path)
    x, y = normalize(data['X'].astype(np.float32), data['Y'].astype(np.float32))
    z = data['Z'].astype(np.float32)
    print(z.shape)

    p = copy.deepcopy(x[:,-12:])
    x_runtime = x[300:301]
    p_runtime = p[300:301]
    z_time = z[300:301]
    np.savez_compressed('./runtime_data.npz', X=x_runtime, P=p_runtime, Z=z_time)
    
    indices = list(range(len(x)))
    test_number = int(test_rate * len(x))
    random.shuffle(indices)
    train, test = indices[test_number:], indices[:test_number]
    train_data = dataloader(batch, x[train], y[train], p[train], z[train])
    test_data = dataloader(batch, x[test], y[test], p[test], z[test])
    
    model = Model(len(x[0]), len(y[0]), hidden, dropout)
    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, model.module.parameters()), lr = learn_rate)
    return model, optimizer, train_data, test_data

def train(model, optimizer, train_data, test_data, epochs, device, gamma = 0.01):
    print('training start')
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        avg_train_loss, avg_test_loss = 0, 0
        model.train()
        for index, batch in enumerate(train_data):
            x, y, p, z = map(lambda x: x.to(device), batch)
            #x_0 = torch.FloatTensor(x[:,0:342])
            #z = torch.FloatTensor(x[:,342:346])
            #print(x.shape)
            optimizer.zero_grad()
            output = model(x, p, z)
            loss = criterion(output, y)
            # print(11111111111)
            # print(p)
            loss.backward()
            optimizer.step()
            avg_train_loss = (avg_train_loss * index + loss.item()) / (index + 1)
            if index % 1 == 0: 
                print(f'epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {avg_test_loss}')
            
        # model.eval()
        with torch.no_grad():
            for index, batch in enumerate(test_data):
                x, y, p, z = map(lambda x: x.to(device), batch)
                output = model(x, p, z)
                loss = criterion(output, y)
                avg_test_loss = (avg_test_loss * index + loss.item()) / (index + 1)

        print(f'epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {avg_test_loss}')
        if epoch % 5 == 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'train_loss': avg_train_loss,
        #         'test_loss': avg_test_loss,
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict() }
        #     filename = os.path.join('training', f'{epoch}.model')
        #     torch.save(checkpoint, filename)
            save_network(model)

""" Function to Save Network Weights """

def save_network(model):

    torch.save(model,"model.pth")

if __name__ == '__main__':
    random.seed(23456)
    torch.manual_seed(23456)
    if torch.cuda.is_available(): torch.cuda.manual_seed(23456)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: " + str(device))
    
    build_path(['training'])
        
    model, optimizer, train_data, test_data = train_prepare(64, 512, 0.7, device, './datasets/database1210_1.npz', 0.0005, 0.1)
    for name in model.state_dict():
        print(name)
    print(model.__dict__)
    print(model.module.w00)
    train(model, optimizer, train_data, test_data, 4000, device)