#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 time_dim,
                 joints_dim,
                 input_channel
                 ):
        super(ConvTemporalGraphical,self).__init__()
        self.hidden_size = 256
        self.feature_size = input_channel*time_dim*joints_dim
        self.elu = nn.ELU()
        self.num_experts1 = 6
        self.num_experts2 = 4
        self.nn_keep_prob = 0.7
        #self.BN = nn.BatchNorm1d(self.feature_size)
        self.gating_layer0 = nn.Linear(self.feature_size,self.hidden_size)
        self.gating_layer1 = nn.Linear(self.hidden_size,self.hidden_size)
        self.gating_layer2 = nn.Linear(self.hidden_size,self.num_experts1)
        self.gating_layer3 = nn.Linear(self.feature_size,self.hidden_size)
        self.gating_layer4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.gating_layer5 = nn.Linear(self.hidden_size,self.num_experts2)


        self.A=nn.Parameter(torch.FloatTensor(self.num_experts1,time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(2))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(self.num_experts2,joints_dim , time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(2))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def get_AT(self, alpha, controlweights, batch_size):
        a = torch.unsqueeze(alpha, 0)  # 4*out*in   -> 4*1*out*in
        a = a.repeat([batch_size, 1, 1, 1,1])  # 4*1*out*in -> 4*?*out*in
        w = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(controlweights, -1), -1)  ,-1)# 4*?        -> 4*?*1*1
        r = w * a  # 4*?*1*1 m 4*?*out*in
        return torch.sum(r, axis=1)  # ?*out*in

    def get_AS(self, beta,controlweights, batch_size):
        b = torch.unsqueeze(beta, 0)  # 4*out*1   -> 4*1*out*1
        b = b.repeat([batch_size, 1, 1, 1,1])  # 4*1*out*1 -> 4*?*out*1
        w = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(controlweights, -1), -1)  ,-1)# 4*?        -> 4*?*1*1
        r = w * b  # 4*?*1*1 m 4*?*out*1
        return torch.sum(r, axis=1)  # ?*out*1


    def forward(self, x):
        Gating_input = x.reshape(-1, self.feature_size)
        h = nn.Dropout(self.nn_keep_prob)(Gating_input)
        h = self.elu(self.gating_layer0(h))

        h = nn.Dropout(self.nn_keep_prob)(h)
        h = self.elu(self.gating_layer1(h))
        h = nn.Dropout(self.nn_keep_prob)(h)
        h = self.gating_layer2(h)
        Gating_weight_s = nn.functional.softmax(h, dim=0)
        t = nn.Dropout(self.nn_keep_prob)(Gating_input)
        t = self.elu(self.gating_layer3(t))
        t = nn.Dropout(self.nn_keep_prob)(t)
        t = self.elu(self.gating_layer4(t))
        t = nn.Dropout(self.nn_keep_prob)(t)
        t = self.gating_layer5(t)
        Gating_weight_t = nn.functional.softmax(t, dim=0)
        batch_size = x.size()[0]
        AT = self.get_AT(self.T,Gating_weight_t,batch_size)
        AS = self.get_AS(self.A,Gating_weight_s,batch_size)
        x = torch.einsum('nctv,nvtq->ncqv', (x, AT))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,ntvw->nctw', (x, AS))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        #return x.contiguous()
        return x.contiguous(),AS,AT




class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):

        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)


        self.gcn=ConvTemporalGraphical(time_dim,joints_dim,in_channels) # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )






        if stride != 1 or in_channels != out_channels:

            self.residual=nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.residual=nn.Identity()


        self.prelu = nn.PReLU()



    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        #x=self.gcn(x)
        x,AS,AT=self.gcn(x)
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        #return x
        return x,AS,AT




class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):

        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1



        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
            ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)]





        self.block=nn.Sequential(*self.block)


    def forward(self, x):

        output= self.block(x)
        return output


# In[11]:


class Model(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):

        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()


        self.st_gcnns.append(ST_GCNN_layer(input_channels,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64,32,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(32,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(64,input_channels,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))


        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))


        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())




    def forward(self, x):
        for gcn in (self.st_gcnns):
            #x = gcn(x)
            x,AS,AT = gcn(x)
            AS = AS.detach().cpu().numpy()[0,0,:]
            AT = AT.detach().cpu().numpy()[0,0,:]
            a = np.mat(AT)
            a= abs(a.reshape(10,10))
            sns.heatmap(a, cmap='Blues')
            plt.show()

        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x=self.prelus[0](self.txcnns[0](x))

        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
        #return x
        return x,AS,AT
