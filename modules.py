from torch import nn
import torch
from torch.distributions import Bernoulli
from torch.distributions.multinomial import Multinomial

thresh = 250
lens = 0.5
decay = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AcFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh)<lens
        return grad_input*temp.float()

act_fun = AcFun.apply

def mem_update(opt, x, mem, spike, decay, recurrent=True):
    if(recurrent):
        mem = mem*decay*(1.-spike) + opt(x)
    else:
        mem = mem*decay + opt(x)
    return mem, act_fun(mem)
#-----------------------------------------
class BaselineNetwork(nn.Module):
    
    def __init__(self, in_planes, out_planes):
        super(BaselineNetwork, self).__init__()
        
        self.fc = nn.Linear(in_planes, out_planes)
        
    def forward(self, x):
        
        """here a problem on b"""
        b = self.fc(x.detach())
        
        return act_fun(b.view(x.size(0),1))


"""
a network that chooses whether or not stop to predict label
input(x): membrane vector
output: int, probability for stopping
"""
class Controller(nn.Module):
    
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()
        
        self.fc = nn.Linear(ninp, nout)
        #self.policy = torch.tensor([0,1,10,100])
    
    def forward(self, x, T, t):
        
        B = x.size(0)

        probs = torch.sigmoid(self.fc(x))
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([.05])
        #m = Multinomial(probs=probs)
        m = Bernoulli(probs=probs)
        action = m.sample()
        log_1 = m.log_prob(1)
        log_p = m.log_prob(0).repeat(1,T).view(B,T,1)
        log_p[:,t,:] = log_1
        
        #multinomial policy!
        #log_p = m.log_prob(action)
        #action *= self.policy
        #action = action.sum(dim=1).view(B,1)
        
        return action.squeeze(0), log_p.squeeze()

class MemPred(nn.Module):
    """
    x -> exp(x)/sum(exp(x))
    """
    def __init__(self):
        super(MemPred, self).__init__()
        self.sft = nn.Softmax(dim=1)
        
    def forward(self, x):
        return x 
        
        
        
        
        
