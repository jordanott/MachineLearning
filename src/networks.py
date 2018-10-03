import math
import torch
import torch.nn as nn

class Memory(torch.nn.Module):
    def __init__(self,num_mem,mem_dim,H=8):
        super(Memory, self).__init__()
        self.H = H
        self.MEM_DIM = mem_dim
        self.M = torch.randn(num_mem,mem_dim)
        self.WQ, self.WK, self.WV = [torch.randn(H,mem_dim//H,mem_dim//H) for _ in range(3)]
    
    def split_heads(self,x):
        """Split x such to add an extra num_heads dimension"""
        return x.view(x.shape[0], self.H, x.shape[1]//self.H).permute(0, 1, 2)

    def project_memory(self,W,M):
        return torch.einsum('hmd,shm->shd',[W,M])
    
    def encode_memory(self,x):
        # concatenat new input to memory matrix
        M_x = torch.cat([self.M,x],dim=0)
        
        # split memory for MHDPA
        M_h_x = self.split_heads(M_x)
        M_h = self.split_heads(self.M)
        
        # MHDPA using new input
        Q = self.project_memory(self.WQ,M_h) 
        K = self.project_memory(self.WK,M_h_x) 
        V = self.project_memory(self.WV,M_h_x) 
        
        _M_ = torch.einsum('sht,thf->shf',[torch.einsum('shf,thf->sht',[Q,K]) / math.sqrt(self.MEM_DIM//self.H), V])
        
        # reshape memories
        _M_ = _M_.contiguous().view(-1,self.MEM_DIM)
        
        return _M_
    
    def forward(self,x):
        return self.encode_memory(x)
