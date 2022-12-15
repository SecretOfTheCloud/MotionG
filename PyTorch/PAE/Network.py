import Library.Utility as utility
import Library.Plotting as plot
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Network as this

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import random

import matplotlib.pyplot as plt

#Start Parameter Section
window = 2.0 #time duration of the time window
frames = 121 #sample count of the time window (60FPS)
keys = 13 #optional, used to rescale the FT window to resolution for motion controller training afterwards
joints = 28

input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-velocity component of each joint)
phase_channels = 6 #desired number of latent phase channels (usually between 2-10)

epochs = 30
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-4

restart_period = 10
restart_mult = 2

plotting_interval = 100 #update visualization at every n-th batch (visualization only)
pca_sequence_count = 10 #number of motion sequences visualized in the PCA (visualization only)
test_sequence_length = 150 #maximum length of each motion sequence (visualization only)
#End Parameter Section

if __name__ == '__main__':
    
    def Item(value):
        return value.detach().cpu()

    data_file = '../../datasets/P/v_p.npz'
    data = np.load(data_file)['vp'].astype(np.float32)
    print(data.shape)

    #Initialize visualization
    sample_count = data.shape[0]
    feature_dim = data.shape[1]
    print(feature_dim)
    #Initialize all seeds
    seed = 23456
    rng = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    #Build network model
    network = utility.ToDevice(this.Model(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        key_range=keys,
        window=window
    ))

    #Setup optimizer and loss function
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(batch_size)
    print(sample_count)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()

    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        rng.shuffle(I)
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]
            # print(11122)
            # print(train_indices)
            #Run model prediction
            # data_batch = utility.ReadBatch(data_file, train_indices, feature_dim)
            data_batch = data[train_indices]
            # print(22222225555555555)
            # print(data[train_indices].shape)
            # print("hhhhhhhhhhh")
            # print(type(data_batch))
            # print(data_batch.shape)
            # print("jjjjjjjjjjjjjjjjjjj")
            train_batch = utility.ToDevice(torch.from_numpy(data_batch))
            # print(111111111111111111111111)
            # print(train_batch.shape)
            # print(222222222222222222222222)
            yPred, latent, signal, params = network(train_batch)

            #Compute loss and train
            loss = loss_function(yPred, train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
          
        print('Epoch', epoch+1)
        torch.save(network.state_dict(), "../../model/DeepPhase.pth")

class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, key_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.key_range = key_range

        self.window = window
        self.time_scale = key_range/time_range

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * (time_range * self.time_scale) / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(input_channels/3)
        
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv2 = nn.BatchNorm1d(num_features=embedding_channels)

        self.fc = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))
            self.bn.append(nn.BatchNorm1d(num_features=2))

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    def atan2(self, y, x):
        tpi = self.tpi
        ans = torch.atan(y/x)
        ans = torch.where( (x<0) * (y>=0), ans+0.5*tpi, ans)
        ans = torch.where( (x<0) * (y<0), ans-0.5*tpi, ans)
        return ans

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        #print(rfft.shape)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        freq = freq / self.time_scale

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component
        # print(freq.shape,amp.shape,offset.shape)
        return freq, amp, offset

    def forward(self, x):
        y = x
        # print(111111111)
        # print(y)
        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.bn_conv1(y)
        y = torch.tanh(y)
        # print(111111111)
        # print(y)

        y = self.conv2(y)
        y = self.bn_conv2(y)
        y = torch.tanh(y)

        latent = y #Save latent for returning
        
        #print(y.shape)
        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            v = self.bn[i](v)
            p[:,i] = self.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning
        print(p)

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        # print("p"+str(p.shape))
        # print("f"+str(f.shape))
        # print("a"+str(a.shape))
        # print("b"+str(b.shape))
        # print("y"+str(y.shape))
        # print("tpi"+str(self.tpi.shape))
        # print("args"+str(self.args.shape))
        # print("args"+str(self.args))






        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.bn_deconv1(y)
        y = torch.tanh(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params