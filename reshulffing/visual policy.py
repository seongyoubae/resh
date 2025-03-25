import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from torch.distributions import Categorical
import random
import numpy as np
import os
import csv

# cfg.py에 정의된 하이퍼파라미터 불러오기
from cfg import get_cfg
# 환경 및 데이터 관련 모듈 (실제 프로젝트에 맞게 준비되어 있어야 함)
from env import Locating
from data import generate_reshuffle_plan, save_reshuffle_plan_to_excel


########################################
# MultiheadSelfAttention 레이어 (PyTorch MultiheadAttention 활용)
########################################
class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_output, attn_weights


########################################
# 가중치 초기화 함수
########################################
def normalized_columns_initializer(tensor, std=1.0):
    nn.init.normal_(tensor, 0, 1)
    with torch.no_grad():
        norm = tensor.pow(2).sum(0, keepdim=True).sqrt()
        tensor *= std / (norm + 1e-8)
    return tensor


########################################
# CombinedAttentionSchedulerNetwork (Actor-Critic with Baseline)
########################################
class CombinedAttentionSchedulerNetwork(nn.Module):
    def __init__(self, meta_data, state_size, num_nodes,
                 embed_dim, num_heads, num_HGT_layers,
                 num_actor_layers, num_critic_layers,
                 parameter_sharing=True,
                 num_pile=4):
        """
        네트워크 구조:
          1. HGTConv 레이어를 통해 그래프 상태 인코딩 (노드 임베딩 생성)
          2. 글로벌 평균 임베딩과 learnable query를 이용해 Multihead Self-Attention을 적용하여
             Attention 기반 컨텍스트를 추출
          3. 글로벌 컨텍스트와 Attention 컨텍스트를 결합하여 최종 상태 표현 생성
          4. Actor 헤드를 통해 두 개의 (source, destination) 액션 분포 생성
          5. Critic 헤드를 통해 상태 가치(= baseline) 예측
        """
        super(CombinedAttentionSchedulerNetwork, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.parameter_sharing = parameter_sharing
        self.num_pile = num_pile

        # 1. 그래프 인코딩: HGTConv 레이어들
        self.conv = nn.ModuleList()
        for i in range(num_HGT_layers):
            in_dim = state_size["plate"] if i == 0 else embed_dim
            self.conv.append(HGTConv(in_channels=in_dim,
                                     out_channels=embed_dim,
                                     metadata=meta_data,
                                     heads=num_heads))

        # 2. 글로벌 요약 및 Attention 쿼리 생성
        self.project_global = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = MultiheadSelfAttention(embed_dim, num_heads)

        # 3. Actor 헤드 (source 및 destination 액션)
        if num_actor_layers == 1:
            self.source_head = nn.Linear(embed_dim, num_pile)
            self.dest_head = nn.Linear(embed_dim, num_pile)
        else:
            actor_layers = []
            input_dim = embed_dim
            for _ in range(num_actor_layers - 1):
                actor_layers.append(nn.Linear(input_dim, embed_dim))
                actor_layers.append(nn.ReLU())
                input_dim = embed_dim
            actor_layers.append(nn.Linear(input_dim, num_pile))
            self.source_head = nn.Sequential(*actor_layers)
            self.dest_head = nn.Sequential(*actor_layers)

        # 4. Critic 헤드 (baseline)
        if num_critic_layers == 1:
            self.critic_linear = nn.Linear(embed_dim, 1)
        else:
            critic_layers = []
            input_dim = embed_dim
            for _ in range(num_critic_layers - 1):
                critic_layers.append(nn.Linear(input_dim, embed_dim))
                critic_layers.append(nn.ReLU())
                input_dim = embed_dim
            critic_layers.append(nn.Linear(input_dim, 1))
            self.critic_linear = nn.Sequential(*critic_layers)

        # 가중치 초기화
        normalized_columns_initializer(self.source_head[0].weight if num_actor_layers > 1 else self.source_head.weight,
                                       std=0.01)
        normalized_columns_initializer(self.dest_head[0].weight if num_actor_layers > 1 else self.dest_head.weight,
                                       std=0.01)
        normalized_columns_initializer(
            self.critic_linear[0].weight if num_critic_layers > 1 else self.critic_linear.weight, std=1.0)

    def forward(self, state):
        """
        state: torch_geometric의 HeteroData 객체
               - state.x_dict: 최소 "plate" key의 피처 텐서
               - state.edge_index_dict: 엣지 인덱스 정보
        """
        x_dict = state.x_dict
        edge_index_dict = state.edge_index_dict

        for conv_layer in self.conv:
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        h_plate = x_dict["plate"]  # (num_nodes, embed_dim)
        if h_plate.size(0) == 0:
            h_plate = torch.zeros((1, self.embed_dim), device=h_plate.device)

        # 글로벌 요약: 모든 노드 임베딩의 평균
        h_global = h_plate.mean(dim=0)  # (embed_dim,)
        global_context = self.project_global(h_global)  # (embed_dim,)

        # Attention 기반 컨텍스트:
        query = self.query_proj(global_context).unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
        key_value = h_plate.unsqueeze(1)  # (num_nodes, 1, embed_dim)
        attn_out, attn_weights = self.attention(key_value)  # (num_nodes, 1, embed_dim)
        attn_context = attn_out.mean(dim=0).squeeze(0)  # (embed_dim,)

        # 결합된 상태 표현
        combined_context = global_context + attn_context  # (embed_dim,)

        # Actor 헤드: source 및 destination 액션 logits → 정책 확률 분포
        source_logits = self.source_head(combined_context)  # (num_pile,)
        dest_logits = self.dest_head(combined_context)  # (num_pile,)
        source_policy = F.softmax(source_logits, dim=-1)
        dest_policy = F.softmax(dest_logits, dim=-1)

        # Critic 헤드: 상태 가치 (baseline)
        value = self.critic_linear(combined_context)  # (1,)

        return (source_policy, dest_policy), value, attn_weights


########################################
# 테스트 실행 코드 (바로 실행 가능)
########################################
if __name__ == "__main__":
    # torch_geometric의 HeteroData를 사용하여 dummy state 생성
    meta_data = (["plate"], [("plate", "blocks", "plate")])
    state_size = {"plate": 3}
    num_nodes = {"plate": 100}

    # 임의의 강판 피처 생성 (100 x 3 텐서)
    x_plate = torch.randn(100, 3)
    # 빈 edge_index 생성 (실제 환경에서는 적절한 edge_index가 필요)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    state = HeteroData()
    state["plate"].x = x_plate
    state["plate", "blocks", "plate"].edge_index = edge_index

    # 네트워크 생성
    net = CombinedAttentionSchedulerNetwork(
        meta_data=meta_data,
        state_size=state_size,
        num_nodes=num_nodes,
        embed_dim=128,
        num_heads=4,
        num_HGT_layers=2,
        num_actor_layers=2,
        num_critic_layers=2,
        parameter_sharing=True,
        num_pile=4
    )

    # 네트워크 forward 실행
    (src_policy, dest_policy), value, attn_weights = net(state)
    print("Source Policy:", src_policy)
    print("Destination Policy:", dest_policy)
    print("State Value (Baseline):", value)
