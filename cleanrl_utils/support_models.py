import torch.nn as nn
import torch


class MsPacmanTransformer(nn.Module):
    def __init__(self, n_actions=9, d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.obs_network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 *7, d_model)
        )
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.positional_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.positional_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward= 4*d_model, 
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self.event_network = nn.Sequential( #aka probability_label_network
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, 13) #, nn.Sigmoid() removed because of using binary_cross_entropy_with_logits in train_transformer.py
        )

        self.distance_network = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, 9)
        )

    def encode(self, obs):
        x = self.obs_network(obs.reshape(-1, *obs.shape[-3:]).float()/255.0)
        return x.reshape(*obs.shape[:-3], -1)

    def head(self, emb, action):
        t = emb.shape[1] #dim 0 is envs
        a = self.action_emb(action.long())
        x = emb + a + self.positional_emb[:, :t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        return torch.cat([self.event_network(x), self.distance_network(x)], dim=-1) #[b, t, labels]
    
    def forward(self, obs, action):
        return self.head(self.encode(obs), action)


class MsPacmanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # => (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # => (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # => (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # => (512)
        )
        self.action_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.fc_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.probability_label_network = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 13),
            nn.Sigmoid()
        )
        self.distance_label_network = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
            )
        
    def forward(self, obs, action):
        obs = obs.float()/255.0
        # Reshape from [64, 4, 4, 84, 84] to [256, 4, 84, 84]
        #obs = obs.view(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))          
        action = action.float().view(-1, 1) #.unsqueeze(1)
        #action = action.view(action.size(0) * action.size(1), 1)
        obs_processed = self.obs_network(obs)
        action_processed = self.action_network(action)
        concat = torch.cat([obs_processed, action_processed], dim=1)
        fc = self.fc_network(concat)
        probabilities = self.probability_label_network(fc)
        distances = self.distance_label_network(fc)
        return torch.cat([probabilities, distances], dim=1)


class FailurePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # => (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # => (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # => (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # => (512)
        )
        self.action_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
   
    def forward(self, obs, action):
        obs = obs.float()/255.0
        action = action.float().view(-1, 1)
        obs_processed = self.obs_network(obs)
        action_processed = self.action_network(action)
        result = self.fc(torch.cat([obs_processed, action_processed], dim=1))
        return torch.sigmoid(result)
