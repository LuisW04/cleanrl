import torch
import numpy as np

class ContextBuffer:
    def __init__(self, model, num_envs, T, device):
        self.model, self.T = model, T
        self.emb = torch.zeros(num_envs, T, model.d_model, device=device)
        self.act = torch.zeros(num_envs, T, dtype=torch.long, device=device)
        self.n = torch.zeros(num_envs, dtype=torch.long, device=device) #current window size per env, 0 <= n <= T
        self.new_action = torch.zeros(num_envs, dtype=torch.long, device=device) 
        self.all_envs = torch.arange(num_envs, device=device)

    @torch.no_grad()
    def step(self, obs):
        full = self.n == self.T
        if full.any(): #Slide the window over by 1
            self.emb[full] = self.emb[full].roll(-1, dims=1)
            self.act[full] = self.act[full].roll(-1, dims=1)
        pos = self.n.clamp(max=self.T-1)
        self.emb[self.all_envs, pos] = self.model.encode(obs)
        self.act[self.all_envs, pos] = self.new_action
        self.n = (self.n + 1).clamp(max=self.T)
        out = self.model.head(self.emb, self.act)
        return out[self.all_envs, self.n -1].sigmoid()

    def update(self, action, done):
        self.new_action = action.clone()
        self.n[done] = 0
        self.new_action[done] = 0

