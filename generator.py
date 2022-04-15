import pickle
import random

def gen(len=100, st=0, en=10):
    ran1 = random.random() * random.randint(-1, 1)
    ran2 = random.random() * random.randint(-1, 1)
    st = st + ran1
    en = en + ran2
    len = len +2
    ll = [None] * len
    ll[0] = st
    ll[-1] = en
    idx = list(range(1,len-1))
    random.shuffle(idx)

    def get_low(tt):
        while ll[tt] is None: tt = tt-1
        return ll[tt]
    def get_upper(tt):
        while ll[tt] is None: tt = tt+1
        return ll[tt]

    for target in idx:
        ll[target] = (get_low(target) + get_upper(target) )/2
    return ll[1:-1]

if __name__ =="__main__":
    class_a = [gen(st=0, en=10) for _ in range(1000)]
    class_b = [gen (st=4, en=14) for _ in range (1000)]
    with open ("class_a", "wb") as fp:  # Pickling
        pickle.dump (class_a, fp)
    with open ("class_b", "wb") as fp:  # Pickling
        pickle.dump (class_b, fp)
        
        
        
 


import pickle
import time

import torch
import numpy as np

from dtw.model import LinearModel, DTWModel
from soft_dtw_cuda import SoftDTW

with open("class_a", "rb") as fp:   # Unpickling
    class_a = pickle.load(fp)
with open("class_b", "rb") as fp:   # Unpickling
    class_b= pickle.load(fp)

Y =[0]*len(class_a) + [1]*len(class_b)
X = class_a + class_b
F = len(class_a[0])
count = len(X)


def dtw():
    x = torch.tensor(X, requires_grad=True)
    x = x.resize(x.size()[0], x.size()[1], 1)
    x = x.cuda ()
    # Create the "criterion" object

    start_tm = time.time()
    sdtw = SoftDTW (use_cuda=True, gamma=0.0001)
    distance = []
    for i in range(count):
        xx = x[i].repeat(count,1,1)
        loss = sdtw (xx, x).cpu().detach().numpy()
        # print(*loss, sep="\t")
        distance.append(loss)
    distance = np.array(distance)
    print("TIME ", time.time() - start_tm)

    iter=10

    summed = np.sum (distance, axis=1)
    summed[summed == 0] = np.nan

    aa = np.nanargmin(summed)
    bb = np.nanargmax(summed)
    if np.sum(X[aa]) > np.sum(X[bb]):
        tt = aa; aa = bb; bb = tt


    pred = np.array([None] * count)
    pred[aa] = 0
    pred[bb] = 1

    def get_near(pre):
        for me in range (count):
            if distance[aa, me] < distance[bb, me]: pre[me] = 0
            else: pre[me] = 1
        return pre

    def get_center(cls):
        _idx = np.argwhere (pred != cls)
        _dist = distance.copy ()
        _dist[_idx] = 0
        _dist[:, _idx] = 0
        summed = np.sum(_dist, axis=1)
        summed[summed==0] = np.nan
        # print(summed)
        # print(np.nanargmin(summed))
        return np.nanargmin(summed)

    for _ in range(iter):

        pred = get_near(pred)
        # print(pred)
        print(np.sum(np.abs(Y-pred)))
        aa = get_center(0)
        bb = get_center(1)


    # print(dist_0.tolist(), sep="\t")

import torch.nn  as nn
import torch as th
def liner(epochs):
    x = torch.tensor (X, requires_grad=True)
    x = x.resize (x.size ()[0], x.size ()[1], 1)
    x = x.cuda ()

    y = torch.tensor (Y, dtype=th.float)
    y = y.resize (y.size ()[0], 1)
    y = y.cuda()


    model = DTWModel((x.size[0], x.size[1])).to("cuda")
    rmse = nn.MSELoss().to('cuda')
    optimizer = th.optim.Adam(model.parameters(), lr=0.01,  weight_decay=1e-4)

    slice = 1000
    step = np.array(range(count))
    np.random.shuffle(step)
    step = step.reshape((slice, -1))

    start = time.time()
    for epoch in range (epochs):
        for batch in step:
            data = x[batch]
            target = y[batch]

            optimizer.zero_grad ()
            hypothesis = model(data)
            loss = rmse (hypothesis, target)
            loss.backward ()
            optimizer.step ()

        pred = model.predict(x)

        acc = np.sum (np.abs ((y - pred).detach().cpu().numpy()))
        print(epoch, ':Loss Eval: {:>.4}, {}'.format(loss, acc))
    # print( pred.detach().cpu().numpy().tolist())
    print('{:>.4}'.format(time.time()-start))

if __name__== "__main__":
    # dtw()
    liner (50)
    
    
    
    
    
    
    
    
   from typing import Tuple

import torch as th
import torch.nn as nn
from torch import Tensor

from soft_dtw_cuda import SoftDTW


class DtwLinear(nn.Module):


    __constants__ = ['in_features', 'out_features']
    indim: Tuple[int]
    outdim: int
    weight: Tensor

    def __init__(self, indim: Tuple[int], outdim:int=None, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DtwLinear, self).__init__()
        self.indim = indim
        seq_len = indim[0]
        feature = indim[1]
        if outdim is None: outdim = seq_len

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



    def __init__(self, in_features: int, out_features: int):
        super(DtwLinear, self).__init__(in_features=in_features, out_features=out_features, bias=False)
        self.sdtw = SoftDTW (use_cuda=True, gamma=0.0001)

    def forward(self, input: Tensor) -> Tensor:
        # print(input.shape)
        print(self.weight.shape)
        return self.sdtw (input, self.weight)



class DTWModel(nn.Module):

    def __init__(self, indim):
        super (LinearModel, self).__init__ ()
        interdim = indim[0]
        self.model = nn.Sequential (
            DtwLinear (indim), nn.Tanh(),
            nn.Linear (interdim, 1), nn.Sigmoid())
        self.pred = nn.Softmax()

    def forward(self, observations: Tensor) -> th.Tensor:
        return  self.model (observations)

    def predict(self, observations: Tensor) -> th.Tensor:
        pred = self.model (observations)
        return th.round(pred)


class LinearModel(nn.Module):

    def __init__(self, indim):
        super (LinearModel, self).__init__ ()
        interdim = int(indim/2)
        self.model = nn.Sequential (
            nn.Linear (indim,1, interdim), nn.Tanh(),
            nn.Flatten(),
            nn.Linear (interdim, 1), nn.Sigmoid())
        self.pred = nn.Softmax()

    def forward(self, observations: Tensor) -> th.Tensor:
        print(observations.shape)
        return  self.model (observations)

    def predict(self, observations: Tensor) -> th.Tensor:
        pred = self.model (observations)
        return th.round(pred)
    
    
   
  
  
  
  
import torch
import numpy as np


if __name__ =="__main__":
    from soft_dtw_cuda import SoftDTW

    # Create the sequences
    batch_size, len_x, len_y, dims = 2, 5, 7, 2
    X = torch.rand ((batch_size, len_x, dims), requires_grad=True)
    Y = torch.rand ((batch_size, len_y, dims))


    X = np.array ([[1, 2, 3, 4, 5], [1, 2, 3, 5, 5]], dtype=float)
    # Time series 2: numpy array, shape = [n, d] where n = length and d = dim
    Y = np.array ([1, 2, 3, 4, 10], dtype=float)
    X=torch.tensor(X, requires_grad=True).resize(2,5,1)
    Y=torch.tensor(Y, requires_grad=True).resize(2,5,1)
    # print(X)

    # Transfer tensors to the GPU
    x = X.cuda ()
    y = Y.cuda ()

    # Create the "criterion" object
    sdtw = SoftDTW (use_cuda=True, gamma=0.1)

    # Compute the loss value
    loss = sdtw (x, y)  # Just like any torch.nn.xyzLoss()

    print(loss)
    # Aggregate and call backward()
    loss.mean ().backward ()

    loss = sdtw (x, y)  # Just like any torch.nn.xyzLoss()

    print(loss)
    # Aggregate and call backward()
    loss.mean ().backward ()
    print (loss)
    
