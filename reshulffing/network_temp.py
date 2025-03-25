import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.nn import HGTConv

def custom_scatter_softmax(src, index, eps=1e-12):
    unique_indices = index.unique(sorted=True)
    out = torch.zeros_like(src)
    for u in unique_indices:
        mask = (index == u)
        group = src[mask]
        max_val = group.max()
        group_exp = torch.exp(group - max_val)
        sum_val = group_exp.sum()
        out[mask] = group_exp / (sum_val + eps)
    return out

def custom_scatter(src, index, eps=1e-12):
    if index.numel() == 0:
        D = src.shape[1]
        return torch.zeros((0, D), device=src.device, dtype=src.dtype)
    unique_indices = index.unique(sorted=True)
    D = src.shape[1]
    out = torch.zeros(index.max().item() + 1, D, device=src.device, dtype=src.dtype)
    for u in unique_indices:
        mask = (index == u)
        out[u] = src[mask].sum(dim=0)
    return out

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.ln(out)
        out = self.act(out)
        return x + out

class SteelPlateAttentionModel(nn.Module):
    """
    확장된 네트워크:
      - HGTConv 기반 그래프 인코더로 입력 상태의 글로벌 컨텍스트를 계산합니다.
      - 다층 네트워크 기반의 joint 액터 헤드를 사용하며, 여기에는 여러 완전 연결 층, LayerNorm, ReLU, Dropout, 그리고 residual block을 포함합니다.
      - 두 개의 크리틱 네트워크(critic1, critic2)를 통해 double-critic 구조로 Q-값 예측을 수행합니다.
      - 향상된 다중헤드 어텐션 메커니즘과 학습 가능한 온도 파라미터로 softmax 분포의 날카로움을 자동 조절합니다
    """

    def __init__(self, state_size, meta_data, embed_dim=128, num_heads=4, num_HGT_layers=2,
                 num_actor_layers=5, num_critic_layers=4, num_source=5, num_dest=5, activation="relu",
                 actor_init_std=0.03, critic_init_std=1.0, use_dropout=True, dropout_rate=0.1):
        super(SteelPlateAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.activation = activation
        self.num_source = num_source
        self.num_dest = num_dest
        self.use_dropout = use_dropout

        # HGTConv layers
        self.conv = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList() if use_dropout else None
        for i in range(num_HGT_layers):
            in_dim = state_size["plate"] if i == 0 else embed_dim
            self.conv.append(HGTConv(in_channels=in_dim,
                                     out_channels=embed_dim,
                                     metadata=meta_data,
                                     heads=num_heads))
            self.layer_norms.append(nn.LayerNorm(embed_dim))
            if use_dropout:
                self.dropouts.append(nn.Dropout(dropout_rate))

        # 기존 gate (어텐션 계산용)
        self.gate = nn.Linear(embed_dim, 1)

        # 향상된 다중헤드 어텐션 게이트
        self.gate_query = nn.Linear(embed_dim, embed_dim)
        self.gate_key = nn.Linear(embed_dim, embed_dim)
        self.gate_value = nn.Linear(embed_dim, embed_dim)
        self.gate_out = nn.Linear(embed_dim, 1)
        self.outbound_proj = nn.Linear(1, embed_dim)
        self.project_global = nn.Linear(embed_dim, embed_dim)

        # 활성화 함수 매핑
        activation_dict = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'tanh': nn.Tanh,
        }

        # 확장된 다층 액터 네트워크
        actor_layers = []
        in_dim = embed_dim
        for i in range(num_actor_layers - 1):
            out_dim = embed_dim * 2 if i == 0 else embed_dim
            actor_layers.append(nn.Linear(in_dim, out_dim))
            actor_layers.append(nn.LayerNorm(out_dim))
            actor_layers.append(activation_dict[activation.lower()]())
            if use_dropout:
                actor_layers.append(nn.Dropout(dropout_rate))
            if in_dim == out_dim:
                actor_layers.append(ResidualBlock(out_dim))
            in_dim = out_dim
        actor_layers.append(nn.Linear(in_dim, num_source * num_dest))
        self.actor_joint = nn.Sequential(*actor_layers)

        # 확장된 다층 크리틱 네트워크 - critic1
        critic_layers1 = []
        in_dim = embed_dim
        for i in range(num_critic_layers - 1):
            critic_layers1.append(nn.Linear(in_dim, embed_dim))
            critic_layers1.append(nn.LayerNorm(embed_dim))
            critic_layers1.append(activation_dict[activation.lower()]())
            if use_dropout:
                critic_layers1.append(nn.Dropout(dropout_rate))
            if in_dim == embed_dim:
                critic_layers1.append(ResidualBlock(embed_dim))
            in_dim = embed_dim
        critic_layers1.append(nn.Linear(in_dim, 1))
        self.critic1 = nn.Sequential(*critic_layers1)

        # 확장된 다층 크리틱 네트워크 - critic2
        critic_layers2 = []
        in_dim = embed_dim
        for i in range(num_critic_layers - 1):
            critic_layers2.append(nn.Linear(in_dim, embed_dim))
            critic_layers2.append(nn.LayerNorm(embed_dim))
            critic_layers2.append(activation_dict[activation.lower()]())
            if use_dropout:
                critic_layers2.append(nn.Dropout(dropout_rate))
            if in_dim == embed_dim:
                critic_layers2.append(ResidualBlock(embed_dim))
            in_dim = embed_dim
        critic_layers2.append(nn.Linear(in_dim, 1))
        self.critic2 = nn.Sequential(*critic_layers2)

        self._init_weights(self.actor_joint, actor_init_std)
        self._init_weights(self.critic1, critic_init_std)
        self._init_weights(self.critic2, critic_init_std)

        # 학습 가능한 온도 파라미터
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def _init_weights(self, module, std):
        if isinstance(module, nn.Sequential):
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    with torch.no_grad():
                        norm = m.weight.pow(2).sum(0, keepdim=True).sqrt()
                        m.weight *= std / (norm + 1e-8)
        else:
            nn.init.normal_(module.weight, 0, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            with torch.no_grad():
                norm = module.weight.pow(2).sum(0, keepdim=True).sqrt()
                module.weight *= std / (norm + 1e-8)

    def forward(self, state):
        if isinstance(state, list):
            state = Batch.from_data_list(state)
        if not isinstance(state, HeteroData):
            raise ValueError("state must be a HeteroData or list of HeteroData objects")

        x = state['plate'].x  # (N, feature_dim)
        if hasattr(state['plate'], 'batch') and state['plate'].batch is not None:
            batch = state['plate'].batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_dict = {'plate': x}
        edge_index = state[('plate', 'blocks', 'plate')].edge_index
        edge_index_dict = {('plate', 'blocks', 'plate'): edge_index}

        # HGTConv layers with residual connection, normalization, activation, dropout
        for i, conv_layer in enumerate(self.conv):
            x_new_dict = conv_layer(x_dict, edge_index_dict)
            if i > 0 and x_dict['plate'].shape == x_new_dict['plate'].shape:
                x_new_dict['plate'] = x_new_dict['plate'] + x_dict['plate']
            x_new_dict['plate'] = self.layer_norms[i](x_new_dict['plate'])
            act_fn = getattr(F, self.activation.lower())
            x_new_dict = {k: act_fn(v) for k, v in x_new_dict.items()}
            if self.use_dropout:
                x_new_dict['plate'] = self.dropouts[i](x_new_dict['plate'])
            x_dict = x_new_dict
        h = x_dict["plate"]

        # 기존 gate를 이용한 어텐션 계산
        gate_scores = self.gate(h)
        outbound = x[:, 0].unsqueeze(1)  # outbound 단일 값
        gate_scores = gate_scores + self.outbound_proj(outbound)
        attn_weights = custom_scatter_softmax(gate_scores, batch)
        pooled = custom_scatter(attn_weights * h, batch)
        global_context = self.project_global(pooled)

        # 향상된 다중헤드 어텐션 메커니즘
        query = self.gate_query(global_context)  # (B, embed_dim)
        key = self.gate_key(h)  # (N, embed_dim)
        value = self.gate_value(h)  # (N, embed_dim)
        attn_weights_list = []
        for b_idx in range(global_context.size(0)):
            b_mask = (batch == b_idx)
            if not torch.any(b_mask):
                continue
            b_query = query[b_idx:b_idx + 1]  # (1, embed_dim)
            b_key = key[b_mask]  # (N_b, embed_dim)
            attn_score = torch.matmul(b_query, b_key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
            outbound_b = x[b_mask, 0:1]  # (N_b, 1)
            outbound_feat = self.outbound_proj(outbound_b)  # (N_b, embed_dim)
            outbound_attn = torch.matmul(b_query, outbound_feat.transpose(-2, -1)) / math.sqrt(self.embed_dim)
            attn_score = attn_score + outbound_attn
            attn_weight = F.softmax(attn_score, dim=-1)  # (1, N_b)
            attn_weights_list.append((b_mask, attn_weight))
        attn_weights_final = torch.zeros((h.size(0), 1), device=h.device)
        for b_mask, weight in attn_weights_list:
            attn_weights_final[b_mask] = weight.transpose(-2, -1)
        weighted_h = h * attn_weights_final
        pooled = custom_scatter(weighted_h, batch)
        global_context = self.project_global(pooled)

        # 액터 및 크리틱 헤드
        joint_logits = self.actor_joint(global_context)  # (B, num_source*num_dest)
        joint_policy = F.softmax(joint_logits, dim=-1)
        # 두 개의 크리틱 네트워크를 통해 Q-값 예측
        value1 = self.critic1(global_context)  # (B, 1)
        value2 = self.critic2(global_context)  # (B, 1)
        return joint_policy, value1, value2, attn_weights_final

    def act(self, state, source_mask, dest_mask, greedy=False):
        joint_policy, value1, value2, attn_weights = self.forward(state)
        # 두 크리틱 중 작은 값을 선택
        value = torch.min(value1, value2)
        # B=1 가정: joint_policy shape -> (num_source*num_dest,)
        joint_policy = joint_policy.squeeze(0)
        joint_mask = torch.ger(source_mask, dest_mask).flatten()  # (num_source*num_dest,)
        joint_logits = torch.log(joint_policy + 1e-10)
        joint_logits = joint_logits.masked_fill(~joint_mask, float('-inf'))
        temperature = self.log_temperature.exp()
        masked_joint_policy = F.softmax(joint_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(masked_joint_policy)
        if greedy:
            joint_index = masked_joint_policy.argmax()
        else:
            joint_index = dist.sample()
        joint_index = joint_index.item()
        action_source = joint_index // self.num_dest
        action_dest = joint_index % self.num_dest
        total_logprob = dist.log_prob(torch.tensor(joint_index, device=joint_policy.device))
        if value.shape[0] == 1:
            value = value.squeeze(0).item()
        else:
            value = value.mean()
        return (action_source, action_dest), total_logprob, value, attn_weights

    def evaluate(self, batch_state, batch_action, batch_source_mask, batch_dest_mask):
        if isinstance(batch_state, list):
            batch_state = Batch.from_data_list(batch_state)
        B = batch_state['plate'].batch.max().item() + 1
        joint_policy, value1, value2, _ = self.forward(batch_state)  # (B, num_source*num_dest)
        # 두 크리틱 중 작은 값을 사용
        value = torch.min(value1, value2)
        joint_action_dim = self.num_source * self.num_dest

        if batch_source_mask.dim() == 1:
            batch_source_mask = batch_source_mask.unsqueeze(0).expand(B, -1)
        if batch_dest_mask.dim() == 1:
            batch_dest_mask = batch_dest_mask.unsqueeze(0).expand(B, -1)

        joint_mask_list = []
        for b in range(B):
            src_mask = batch_source_mask[b]
            if src_mask.dim() == 0:
                src_mask = src_mask.unsqueeze(0)
            dst_mask = batch_dest_mask[b]
            if dst_mask.dim() == 0:
                dst_mask = dst_mask.unsqueeze(0)
            mask = torch.ger(src_mask, dst_mask).flatten()
            if mask.numel() != joint_action_dim:
                raise ValueError(f"Batch {b}: Expected joint mask size {joint_action_dim}, got {mask.numel()}")
            joint_mask_list.append(mask)
        joint_mask = torch.stack(joint_mask_list, dim=0)  # (B, joint_action_dim)

        masked_joint_logits = torch.log(joint_policy + 1e-10)
        penalty = -1e6
        masked_joint_logits = masked_joint_logits + (~joint_mask) * penalty
        masked_joint_policy = F.softmax(masked_joint_logits, dim=-1)
        dist = torch.distributions.Categorical(masked_joint_policy)
        joint_actions = batch_action[:, 0] * self.num_dest + batch_action[:, 1]
        joint_actions = joint_actions.unsqueeze(1)  # (B, 1)
        log_prob = torch.gather(torch.log(masked_joint_policy + 1e-10), 1, joint_actions)
        total_log_prob = log_prob
        dist_entropy = dist.entropy().unsqueeze(1)
        return total_log_prob, value, dist_entropy

    def print_joint_softmax(self, state, source_mask, dest_mask):
        joint_policy, _, _, _ = self.forward(state)
        joint_policy = joint_policy.squeeze(0)
        joint_mask = torch.ger(source_mask, dest_mask).flatten()
        penalty = -1e6
        joint_logits = torch.log(joint_policy + 1e-10)
        joint_logits = joint_logits + (~joint_mask) * penalty
        masked_joint_policy = F.softmax(joint_logits, dim=-1)
        print("Joint softmax probability distribution:")
        print(masked_joint_policy)
