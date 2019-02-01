import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf


import stl.stl as stl
import IPython
import sympy as sym
from stl_main import *
from data_collection import *
from features import *
import utils




import sys
sys.path.insert(0, '/home/karenleung/repos/trams')
import os

import stl.stl as stl

PI = 3.141592653589793

def get_mean_var_numpy(data, tl):
#     IPython.embed(banner1='3453')
    data_ = data.reshape([-1, data.shape[-1]])
    μ = np.sum(data_, axis=0)/np.sum(tl)
    σ = np.sqrt(np.sum(data_**2 - 2 * data_ * μ, axis=0)/np.sum(tl) + μ**2)
    return μ, σ

def get_mean_var_pytorch(data, tl, extract_dict):
    if extract_dict == 'slim':
        int_index = range(28)              # slim_dict
    else:
        int_index = [6, 8 ,15, 17, 24, 26, 33, 35, 42, 44, 51, 53, 60, 62]              # default_dict
    data_reshape = torch.reshape(data, [-1, data.shape[-1]])
    data_mean = torch.sum(data_reshape, dim=0)/torch.sum(tl).type(torch.float)
    # data_mean[int_index] = 0 

    A = torch.sum(data_reshape**2 - 2 * data_reshape * data_mean, dim=0)/(torch.sum(tl.type(torch.float)))
    data_std = torch.sqrt(A + data_mean**2) 
    
    # data_std = (data - data_mean)**2/(torch.sum(tl.type(torch.float))-1)
    # sd = torch.cumsum(data_std, 1)
    index = torch.unsqueeze(tl-1,1)
    index = index.unsqueeze(2)
    index = index.repeat(1, 1, data.shape[-1])
    # data_std = torch.sqrt(torch.sum(torch.gather(sd, 1, index.type(torch.int64)).squeeze(), dim=0))
    # data_std[int_index] = 1
#     IPython.embed(banner1='dfgdf')
    data_std[torch.isnan(data_std)] = torch.zeros_like(data_std[torch.isnan(data_std)] )
    data_std += torch.rand(data_std.shape)*0.01
    return data_mean, data_std

class LaneswapDatasetCol(torch.utils.data.Dataset):
    def __init__(self, data_np, tl, meanvar=(None, None), mhl=5, scale=10):
        self.data_np = data_np    # [batch_size, time, state]
        self.tl = tl              # [batch_size,]
        self.mhl = mhl
        self.mean = meanvar[0]
        self.var = meanvar[1]
        self.scale = scale
        
    def __len__(self):
        return len(self.tl)
    
    def __getitem__(self, idx):
        tl = self.tl[idx].clamp(self.mhl, 120)
        timestep = int(self.mhl - 1 + np.mod(np.random.randint(0, 2**31-1), tl-self.mhl + 1))
        data = self.data_np[idx,:timestep+1,:]
        ttc = np.ceil((tl - timestep)/self.scale).type(torch.int)
        ttc = min(ttc, torch.tensor(2, dtype=torch.int))
        # if ttc < 1:
        #     print("TTC", ttc)
        # ttc = max(ttc, torch.tensor(1, dtype=torch.int))
            # ttc = max(min(ttc, torch.tensor(10, dtype=torch.int)), torch.tensor(1, dtype=torch.int))
        data = self.data_np[idx,:,:]
        if self.mean is not None:
            data -= self.mean
            data /= self.var
        return {'data': data, 'tl': timestep, 'tl_max': tl}, ttc

        # return {

class LaneswapDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, tl, col, meanvar=(None, None), mhl=5, scale=10):
        self.data_np = data_np    # [batch_size, time, state]
        self.tl = tl              # [batch_size,]
        self.col = col            # [batch_size,]
        self.mhl = mhl
        self.mean = meanvar[0]
        self.var = meanvar[1]
        self.scale = scale
        
    def __len__(self):
        return len(self.tl)
    
    def __getitem__(self, idx):

        tl = self.tl[idx].clamp(self.mhl, 120)
        col = self.col[idx]
        timestep = int(self.mhl - 1 + np.mod(np.random.randint(0, 2**31-1), tl-self.mhl + 1))
        data = self.data_np[idx,:tl-1,:]
        if not col:
            ttc = tl/self.scale
        else:
            ttc = np.ceil((tl - timestep)/self.scale).type(torch.int)
            ttc = max(ttc, torch.tensor(1, dtype=torch.int))
            # ttc = max(min(ttc, torch.tensor(10, dtype=torch.int)), torch.tensor(1, dtype=torch.int))
        data = self.data_np[idx,:,:]
        if self.mean is not None:
            data -= self.mean
            data /= self.var
        return {'data': data, 'tl': int(tl), 'tl_max': tl, 'collision': col}, col

        # return {'data': data, 'tl': timestep, 'tl_max': tl, 'collision': col}, ttc.type(torch.float)/120*self.scale + torch.rand(ttc.shape)*0.1



def sigmoid_anneal(start, finish, center_step, steps_lo_to_hi, step):
    return start + (finish - start)*torch.nn.Sigmoid()(torch.tensor(step - center_step, dtype=torch.float) * torch.tensor(1/steps_lo_to_hi, dtype=torch.float))

class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.rnn_q = torch.nn.LSTM(input_size=self.params.model.input_dim+1, 
                                   hidden_size = self.params.rnn.dim,
                                   num_layers=self.params.rnn.nlayers,
                                   bias=True,
                                   batch_first=True,
                                   dropout=0.1,
                                   bidirectional=False)
        self.rnn_p = torch.nn.LSTM(input_size=self.params.model.input_dim, 
                                   hidden_size = self.params.rnn.dim,
                                   num_layers=self.params.rnn.nlayers,
                                   bias=True,
                                   batch_first=True,
                                   dropout=0.1,
                                   bidirectional=False)
        self.mu_projection = torch.nn.Linear(self.params.rnn.dim, self.params.model.latent_dim)
        self.log_sigma_projection = torch.nn.Linear(self.params.rnn.dim, self.params.model.latent_dim)
        self.p_z_x_dist = None
        self.q_z_xy_dist = None


    def q_z_xy(self, x, y):
        rnn_input = x['data']               # [batch_size, time_max, state_dim]
        tl = x['tl']                        # [batch_size]  
        # y is [batch_size]
        # y_ = y.unsqueeze(1).unsqueeze(2).repeat(1, rnn_input.shape[1], 1).type(torch.float32)
        xy = torch.cat([rnn_input, y.unsqueeze(1).unsqueeze(2).repeat(1, rnn_input.shape[1], 1).type(torch.float32)], dim=2)       # [batch_size, time_max, state_dim+ output_dim]
        outputs, _ = self.rnn_q(xy)
        index = torch.unsqueeze(tl-1,1)
        index = index.unsqueeze(2)
        index = index.repeat(1, 1, self.params.rnn.dim)
        rnn_output = torch.gather(outputs, 1, index).squeeze()
        mu = self.mu_projection(rnn_output)
        log_sigma = self.log_sigma_projection(rnn_output)
        sigma = torch.exp(torch.clamp(log_sigma, -self.params.model.log_sigma_clamp, self.params.model.log_sigma_clamp))
        self.q_z_xy_dist = torch.distributions.normal.Normal(mu, sigma)
        return mu, sigma

    def p_z_x(self, x):
        # typically this is a normal (0,1)
        rnn_input = x['data']               # [batch_size, time_max, state_dim]
        tl = x['tl']                        # [batch_size]  ]
        outputs, _ = self.rnn_p(rnn_input)
        index = torch.unsqueeze(tl-1,1)
        index = index.unsqueeze(2)
        index = index.repeat(1, 1, self.params.rnn.dim)
        rnn_output = torch.gather(outputs, 1, index).squeeze()
        mu = self.mu_projection(rnn_output)
        log_sigma = self.log_sigma_projection(rnn_output)
        sigma = torch.exp(torch.clamp(log_sigma, -self.params.model.log_sigma_clamp, self.params.model.log_sigma_clamp))
        self.p_z_x_dist = torch.distributions.normal.Normal(mu, sigma)
        if self.params.model.standard_p:
            return torch.zeros_like(mu), torch.ones_like(sigma)
        return mu, sigma

    def forward(self, x, y, mode):
        mu_p, sigma_p = self.p_z_x(x)
        mu_q, sigma_q = self.q_z_xy(x, y)
        if mode == 'train':
            mu_q, sigma_q
            return mu_q, sigma_q
        elif mode == 'eval':
            return mu_p, sigma_p
        else:
            raise NameError(mode)


            
class Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.fc_z = torch.nn.Sequential(torch.nn.Linear(self.params.model.latent_dim, self.params.model.fc_dim),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(self.params.model.fc_dim, self.params.model.output_dim),
                                        torch.nn.Sigmoid(),
                                        # torch.nn.Linear(self.params.model.fc_dim, self.params.model.output_dim)
                                        )

        # self.rnn = torch.nn.LSTM(input_size=self.params.model.input_dim, 
        #                          hidden_size = self.params.rnn.dim,
        #                          num_layers=self.params.rnn.nlayers,
        #                          bias=True,
        #                          batch_first=True,
        #                          dropout=0.1,
        #                          bidirectional=False)
    def forward(self, z):
        # rnn_input = x['data']               # [batch_size, time_max, state_dim]
        # tl = x['tl']                        # [batch_size]  
        # outputs, _ = self.rnn_q(rnn_input)
        # index = torch.unsqueeze(tl-1,1)
        # index = index.unsqueeze(2)
        # index = index.repeat(1, 1, self.params.rnn.dim)
        # rnn_output = torch.gather(outputs, 1, index).squeeze()      # [batch_size, rnn_dim]
        return self.fc_z(z)
        
        
class LaneswapModel(torch.nn.Module):
    def __init__(self, params):
        super(LaneswapModel, self).__init__()
        self.params = params
        torch.manual_seed(self.params.learning.seed)
        torch.cuda.manual_seed(self.params.learning.seed)
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def anneal_parameters(self, step):
        start, finish, crossover, divisor = self.params.learning.kl_anneal
        self.kl_weight = torch.tensor(sigmoid_anneal(start,finish, crossover, crossover/divisor, step), dtype=torch.float)

    def reparameterize(self, mu, sigma):
        if self.params.settings.cuda:
            std_z = torch.randn(size=sigma.size(), dtype=torch.float).cuda()
        else:
            std_z = torch.randn(size=sigma.size(), dtype=torch.float)
        return mu + sigma * std_z

    def train_loss(self, log_p_y, y):
        index = y
        y_one_hot = torch.zeros_like(y).unsqueeze(1).repeat(1, self.params.model.output_dim).scatter(1, index.unsqueeze(1).type(torch.int64), 1).type(torch.float32)
        # print(log_p_y.shape, y_one_hot.shape)
        # print(torch.sum(log_p_y*y_one_hot, dim=1).shape)
        ll = torch.mean(torch.sum(log_p_y*y_one_hot, dim=1))
        mu_q, sigma_q = self.encoder.q_z_xy_dist.mean, self.encoder.q_z_xy_dist.variance
        mu_p, sigma_p = self.encoder.p_z_x_dist.mean, self.encoder.p_z_x_dist.variance
        det_sigma_q = torch.prod(sigma_q, dim=1)
        det_sigma_p = torch.prod(sigma_p, dim=1)
        kl = torch.mean(0.5 * (torch.log(det_sigma_p) - torch.log(det_sigma_q) +  torch.sum((mu_q - mu_p)**2 * sigma_p, dim=1) +  torch.sum(sigma_q / sigma_p, dim=1) - self.params.model.latent_dim))
        return torch.mean(- ll + self.kl_weight*kl), ll, kl

    def forward(self, x, y, mode):
        mu, sigma = self.encoder(x, y, mode)

        # will be from q_z_xy or p_z_x depending on the mode
        z = self.reparameterize(mu, sigma)
        p_y = self.decoder(z).clamp(-5,5)
        # pis = torch.exp(log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True))
        log_p_y = p_y - torch.logsumexp(p_y, dim=-1, keepdim=True)
        if mode == 'train':
            return log_p_y, self.train_loss(log_p_y, y)
        elif mode == 'eval':
            return log_p_y, self.train_loss(log_p_y, y)
        else:
            raise NameError('mode')


class LaneswapModelGaussianPrior(torch.nn.Module):
    def __init__(self, params):
        super(LaneswapModelGaussianPrior, self).__init__()
        self.params = params
        torch.manual_seed(self.params.learning.seed)
        torch.cuda.manual_seed(self.params.learning.seed)
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.log_pis_projection = torch.nn.Linear(self.params.model.output_dim, self.params.model.gmm_components)
        self.mus_projection = torch.nn.Linear(self.params.model.output_dim, self.params.model.gmm_components)
        self.log_sigmas_projection = torch.nn.Linear(self.params.model.output_dim, self.params.model.gmm_components)
        self.kl_weight = 0.0

    def anneal_parameters(self, step):
        start, finish, crossover, divisor = self.params.learning.kl_anneal
        self.kl_weight = torch.tensor(sigmoid_anneal(start,finish, crossover, crossover/divisor, step), dtype=torch.float)

    def reparameterize(self, mu, sigma):
        # std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if self.params.settings.cuda:
            std_z = torch.randn(size=sigma.size(), dtype=torch.float).cuda()
        else:
            std_z = torch.randn(size=sigma.size(), dtype=torch.float)
        return mu + sigma * std_z

    def train_loss(self, y, pis, mus, sigmas):
        # ll = -(torch.mean(torch.log(sigma) + (mu - y.type(torch.float))**2/2/sigma**2))
        # ll = -torch.mean((mus - y.type(torch.float))**2/2)
        # ll = torch.mean(torch.log(torch.sum(pis * torch.exp(-(mus - y.unsqueeze(1).type(torch.float))**2/2/sigmas**2)/(2*PI), dim=1)))
        ll = torch.mean(torch.sum( (torch.log(pis) - (mus - y.unsqueeze(1).type(torch.float))**2/2/sigmas**2 - 0.5 * 2 * np.pi*sigmas), dim=1))
        # ll = torch.mean(torch.sum(torch.log(pis * torch.exp(-(mus - y.unsqueeze(1).type(torch.float))**2/2/sigmas**2)/(2*PI)), dim=1))
        mu_q, sigma_q = self.encoder.q_z_xy_dist.mean, self.encoder.q_z_xy_dist.variance
        mu_p, sigma_p = self.encoder.p_z_x_dist.mean, self.encoder.p_z_x_dist.variance
        det_sigma_q = torch.prod(sigma_q, dim=1)
        det_sigma_p = torch.prod(sigma_p, dim=1)
        kl = torch.mean(0.5 * (torch.log(det_sigma_p) - torch.log(det_sigma_q) +  torch.sum((mu_q - mu_p)**2 * sigma_p, dim=1) +  torch.sum(sigma_q / sigma_p, dim=1) - self.params.model.latent_dim))
        return torch.mean(- ll + self.kl_weight*kl), ll, kl

    def eval_loss(self, y, pis, mus, sigmas):
        _, ll, kl = self.train_loss(y, pis, mus, sigmas)
        return torch.mean(- ll + kl), ll, kl


    def forward(self, x, y, mode):
        mu, sigma = self.encoder(x, y, mode)
        # will be from q_z_xy or p_z_x depending on the mode
        z = self.reparameterize(mu, sigma)
        p_y = self.decoder(z)
        log_pis = torch.tanh(self.log_pis_projection(p_y))
        pis = torch.exp(log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True))
        mus = self.params.model.y_max * torch.sigmoid(self.mus_projection(p_y))
        log_sigmas = self.log_sigmas_projection(p_y).clamp(-self.params.model.log_sigma_clamp, self.params.model.log_sigma_clamp)
        sigmas = torch.exp(log_sigmas)

        if mode == 'train':
            return pis, mus, sigmas, self.train_loss(y, pis, mus, sigmas)
        elif mode == 'eval':
            return pis, mus, sigmas, self.eval_loss(y, pis, mus, sigmas)
        else:
            raise NameError('mode')



class LaneswapModelLogistic(torch.nn.Module):
    def __init__(self, params):
        super(LaneswapModelLogistic, self).__init__()
        self.params = params
        torch.manual_seed(self.params.learning.seed)
        torch.cuda.manual_seed(self.params.learning.seed)
        self.encoder = Encoder(params)
        self.rnn = torch.nn.LSTM(input_size=self.params.model.input_dim, 
                                 hidden_size = self.params.rnn.dim,
                                 num_layers=self.params.rnn.nlayers,
                                 bias=True,
                                 batch_first=True,
                                 dropout=0.1,
                                 bidirectional=False)
        # self.decoder = Decoder(params)
        # self.kl_weight = 0.0
        self.logistic_projection = torch.nn.Linear(self.params.rnn.dim, 1)

    # def anneal_parameters(self, step):
    #     start, finish, crossover, divisor = self.params.learning.kl_anneal
    #     self.kl_weight = torch.tensor(sigmoid_anneal(start,finish, crossover, crossover/divisor, step), dtype=torch.float)

    # def reparameterize(self, mu, sigma):
    #     if self.params.settings.cuda:
    #         std_z = torch.randn(size=sigma.size(), dtype=torch.float).cuda()
    #     else:
    #         std_z = torch.randn(size=sigma.size(), dtype=torch.float)
    #     return mu + sigma * std_z
    def rnn_encoder(self, x):
        # typically this is a normal (0,1)
        rnn_input = x['data']               # [batch_size, time_max, state_dim]
        tl = x['tl']                        # [batch_size]  ]
        outputs, _ = self.rnn(rnn_input)
        index = torch.unsqueeze(tl-1,1)
        index = index.unsqueeze(2)
        index = index.repeat(1, 1, self.params.rnn.dim)
        return torch.gather(outputs, 1, index).squeeze()

    def train_loss(self, y, c):
        loss = torch.mean(y.type(torch.float)*torch.log(10*c) + (1- y.type(torch.float))*torch.log((1 - c)))
        ll = torch.mean(y.type(torch.float)*torch.log(c) + (1- y.type(torch.float))*torch.log((1 - c)))
        # mu_q, sigma_q = self.encoder.q_z_xy_dist.mean, self.encoder.q_z_xy_dist.variance
        # mu_p, sigma_p = self.encoder.p_z_x_dist.mean, self.encoder.p_z_x_dist.variance
        # det_sigma_q = torch.prod(sigma_q, dim=1)
        # det_sigma_p = torch.prod(sigma_p, dim=1)
        # kl = torch.mean(0.5 * (torch.log(det_sigma_p) - torch.log(det_sigma_q) +  torch.sum((mu_q - mu_p)**2 * sigma_p, dim=1) +  torch.sum(sigma_q / sigma_p, dim=1) - self.params.model.latent_dim))
        return -loss, ll,

    def eval_loss(self, y, c):
        _, ll, = self.train_loss(y, c)
        return -ll , ll, 


    def forward(self, x, y, mode):
        rnn_output = self.rnn_encoder(x)
        c = torch.sigmoid(self.logistic_projection(rnn_output))
        if mode == 'train':
            return c, self.train_loss(y, c)
        elif mode == 'eval':
            return c, self.eval_loss(y, c)
        else:
            raise NameError('mode')

