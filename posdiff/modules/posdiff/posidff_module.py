import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


def knn(x, k):
    """get k nearest neighbors based on distance in feature space
    Args:
        x: [b,dims(=3),num]
        k: number of neighbors to select

    Returns:
        k nearest neighbors (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]

    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num]

    pairwise_distance = -xx - inner
    pairwise_distance = pairwise_distance - xx.transpose(2, 1).contiguous()  # [b,num,num]
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def get_graph_feature_position(x, k=16, idx=None):
    batch_size, dims, num_points = x.size()
    k = min(num_points, k)-1
    #print('dims', dims)
    x = x.view(batch_size, -1, num_points)
    x_position = x[:,int(dims/2):int(dims),:]
    if idx is None:
        #idx = knn(x, k=k)  # (batch_size, num_points, k)
        idx = knn(x_position, k=k)  # (batch_size, num_points, k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx=idx.to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]
    idx = idx + idx_base  # (batch_size, num_points, k)

    idx = idx.view(-1)  # (batch_size * num_points * k)
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,dims)
    feature = feature.view(batch_size, num_points, k, dims)  # (batch_size, num_points, k, dims)
    x = x.view(batch_size, num_points, 1, dims).repeat(1, 1, k, 1)  # [B, num, k, dims]


    feature_point = torch.cat((feature[:,:,:,0:int(dims/2)], x[:,:,:,0:int(dims/2)]), dim=3).permute(0, 3, 1, 2)  # [B, dims*2, num, k]
    feature_position = torch.cat((feature[:,:,:,int(dims/2):int(dims)], x[:,:,:,int(dims/2):int(dims)]), dim=3).permute(0, 3, 1, 2)  # [B, dims*2, num, k]
    feature = torch.cat((feature_point, feature_position), dim=1) 


    return feature


class DGCNN_func(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN_func, self).__init__()
        
        self.emb_dims = emb_dims
        #self.conv1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(2*self.emb_dims, self.emb_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(2*self.emb_dims+self.emb_dims, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.emb_dims)
        self.bn2 = nn.BatchNorm2d(self.emb_dims)
        
    def forward(self, t, x_input):
    #def forward(self, t, x):
        batch_size, num_dims, num_points = x_input.size()  
        
        x = get_graph_feature_position(x_input) 

        
        x0 = x.max(dim=-1, keepdim=True)[0]  

        x = F.relu(self.bn1(self.conv1(x)))  
        x1 = x.max(dim=-1, keepdim=True)[0]  

        x = torch.cat((x0, x1), dim=1)  

        x = F.relu(self.bn2(self.conv2(x))).view(batch_size, -1, num_points)  

        return x 

class PosDiffNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PosDiffNet, self).__init__()
        self.emb_dims = emb_dims
        self.odeint = odeint
        
        tol_scale = torch.tensor([1.0], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.atol = tol_scale * 1e-2
        self.rtol = tol_scale * 1e-2
        
        self.odefunc = DGCNN_func(emb_dims=self.emb_dims)
        
        self.method = 'euler'
        self.step_size = 1.0
        self.t = torch.tensor([0, 1.0], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        
        t = self.t.type_as(x) 
        integrator = self.odeint
        func = self.odefunc
        state = x
        state_dt = integrator(
            func, state, t,
            method=self.method,
            options={'step_size': self.step_size},
            atol=self.atol,
            rtol=self.rtol)
        z_out = state_dt[1]
         
        return z_out
