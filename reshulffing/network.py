import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#####################################
# 모델에 필요한 상수들 (기존 인터페이스와 호환)
#####################################
MAX_SOURCE = 5     # num_from_piles
MAX_DEST = 5       # num_to_piles

def init_weights(model, init_std):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def build_mlp(input_dim, output_dim, hidden_dim, num_layers, activation=nn.ELU, use_dropout=False, use_layernorm=False):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_layernorm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    if use_dropout:
        layers.append(nn.Dropout(p=0.1))
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 target_entropy=1.0,
                 num_actor_layers=2,
                 num_critic_layers=2,
                 actor_init_std=0.01,
                 critic_init_std=0.5,
                 max_stack=30,
                 num_from_piles=MAX_SOURCE,
                 num_to_piles=MAX_DEST):
        """
        각 pile의 상태(상위 max_stack개의 outbound 값과 평균 outbound, 즉 pile_feature_dim = max_stack+1)를
        개별적으로 임베딩한 후 self-attention으로 전체 pile 간 상호작용을 학습합니다.
        첫 num_from_piles는 source piles, 나머지 num_to_piles는 destination piles에 해당합니다.
        """
        super(SteelPlateConditionalMLPModel, self).__init__()
        self.max_stack = max_stack
        self.pile_feature_dim = max_stack + 1  # 각 pile의 feature 차원
        self.num_from_piles = num_from_piles
        self.num_to_piles = num_to_piles
        self.total_piles = num_from_piles + num_to_piles

        # 각 pile feature를 embed_dim 차원으로 임베딩
        self.pile_encoder = nn.Sequential(
            nn.Linear(self.pile_feature_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 전체 pile 임베딩([B, total_piles, embed_dim])에 대해 self-attention 적용
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)

        # Actor: global representation으로부터 source logits 산출
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.num_from_piles)
        )
        init_weights(self.actor_head, actor_init_std)

        # Critic: global representation으로부터 가치 추정
        self.critic_net = build_mlp(embed_dim, 1, embed_dim, num_critic_layers, nn.GELU,
                                    use_dropout=False, use_layernorm=True)
        init_weights(self.critic_net, critic_init_std)

        # Destination 선택: conditional attention 사용
        self.dest_embeddings = nn.Parameter(torch.randn(self.num_to_piles, embed_dim))
        self.dest_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.dest_fc = nn.Linear(embed_dim, self.num_to_piles)
        init_weights(self.dest_fc, actor_init_std)

    def forward(self, pile_features, debug=False):
        """
        입력:
            pile_features: [B, total_piles, pile_feature_dim]
                - 앞쪽 num_from_piles는 source piles, 뒤쪽 num_to_piles는 destination piles.
        출력:
            source_logits: [B, num_from_piles] - raw logits for source piles
            dest_logits: [B, num_to_piles] - raw logits for destination piles
            value: [B, 1]
            global_repr: [B, embed_dim] (최종 global representation)
        """
        # 1. 각 pile feature 임베딩
        pile_emb = self.pile_encoder(pile_features)  # [B, total_piles, embed_dim]

        # 2. Self-attention으로 상호작용 (출력과 attention weight 둘 다 획득)
        attn_output, attn_weights = self.self_attention(pile_emb, pile_emb, pile_emb)
        # 기본 global representation: 간단 평균(pooling)
        global_repr_avg = attn_output.mean(dim=1)  # [B, embed_dim]

        # 3. Attention weight를 활용한 가중치 표현 계산
        # attn_weights: [B, total_piles, total_piles] (각 query별 key의 attention)
        # 모든 query의 가중치를 평균내어 각 pile의 중요도를 도출 (예: [B, total_piles])
        importance = attn_weights.mean(dim=1)
        importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-10)
        # weighted sum: 각 pile 임베딩에 importance를 곱한 후 합산
        global_repr_weighted = (pile_emb * importance.unsqueeze(-1)).sum(dim=1)

        # 4. 두 표현을 간단히 혼합: (선형 결합 혹은 concat 후 차원 조정 가능)
        alpha = 0.5  # 두 표현의 가중치 (하이퍼파라미터로 조정 가능)
        global_repr = alpha * global_repr_avg + (1 - alpha) * global_repr_weighted

        # 5. Actor: source logits 산출 (독립적으로)
        source_logits = self.actor_head(global_repr)  # [B, num_from_piles]

        # 6. Destination 선택: conditional attention 사용
        query = global_repr.unsqueeze(1)  # [B, 1, embed_dim]
        dest_emb_expanded = self.dest_embeddings.unsqueeze(0).expand(global_repr.size(0), -1, -1)
        attn_output_dest, _ = self.dest_attention(query, dest_emb_expanded, dest_emb_expanded)
        attn_output_dest = attn_output_dest.squeeze(1)
        dest_logits = self.dest_fc(attn_output_dest)  # [B, num_to_piles]

        # 7. Critic: 가치 예측
        value = self.critic_net(global_repr)  # [B, 1]

        if debug:
            print("Global representation shape:", global_repr.shape)
            print("Source logits shape:", source_logits.shape)
            print("Destination logits shape:", dest_logits.shape)

        return source_logits, dest_logits, value, global_repr

    @torch.no_grad()
    def act_batch(self, pile_features, source_masks=None, dest_masks=None, greedy=False, debug=False):
        """
        입력:
            pile_features: [B, total_piles, pile_feature_dim]
            source_masks: [B, num_from_piles] (bool tensor; 유효한 source 액션만 True)
            dest_masks: [B, num_to_piles] (bool tensor; 유효한 destination 액션만 True)
        출력:
            actions: [B, 2] (각 행: [selected_source, selected_dest])
            joint_logprob: [B] (선택된 행동의 로그 확률 합)
            value: [B] (critic의 가치 예측)
            global_repr: [B, embed_dim]
        """
        self.eval()
        B = pile_features.size(0)
        source_logits, dest_logits, value, global_repr = self.forward(pile_features, debug=debug)

        # logits에 직접 마스킹: 유효하지 않은 액션에는 매우 낮은 값(-1e9) 대입
        if source_masks is not None:
            masked_source_logits = source_logits.masked_fill(~source_masks, -1e9)
        else:
            masked_source_logits = source_logits

        if dest_masks is not None:
            masked_dest_logits = dest_logits.masked_fill(~dest_masks, -1e9)
        else:
            masked_dest_logits = dest_logits

        # 마스킹된 logits에 softmax 적용 → 독립적인 확률 분포 계산
        source_policy = F.softmax(masked_source_logits, dim=-1)
        dest_policy = F.softmax(masked_dest_logits, dim=-1)

        # 행동 선택: greedy 또는 샘플링
        if greedy:
            selected_source = source_policy.argmax(dim=-1)
            selected_dest = dest_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(source_policy, 1).squeeze(-1)
            selected_dest = torch.multinomial(dest_policy, 1).squeeze(-1)

        # 로그 확률 계산
        chosen_src_logprob = torch.log(source_policy.gather(1, selected_source.unsqueeze(1)) + 1e-10).squeeze(1)
        chosen_dest_logprob = torch.log(dest_policy.gather(1, selected_dest.unsqueeze(1)) + 1e-10).squeeze(1)
        joint_logprob = chosen_src_logprob + chosen_dest_logprob

        actions = torch.stack([selected_source, selected_dest], dim=-1)
        self.last_source_probs = source_policy.detach().clone()
        self.last_dest_probs = dest_policy.detach().clone()

        if debug:
            print("[DEBUG act_batch] source_logits:", source_logits)
            print("[DEBUG act_batch] dest_logits:", dest_logits)
            print("[DEBUG act_batch] masked_source_logits:", masked_source_logits)
            print("[DEBUG act_batch] masked_dest_logits:", masked_dest_logits)
            print("[DEBUG act_batch] source_policy:", source_policy)
            print("[DEBUG act_batch] dest_policy:", dest_policy)
            print("[DEBUG act_batch] selected_source:", selected_source)
            print("[DEBUG act_batch] selected_dest:", selected_dest)
            print("[DEBUG act_batch] actions:", actions)
            print("[DEBUG act_batch] joint_logprob:", joint_logprob)

        return actions, joint_logprob, value.squeeze(-1), global_repr

    def evaluate(self, batch_pile_features, batch_source_mask, batch_dest_mask, batch_action):
        """
        평가용 메서드:
            batch_pile_features: [B, total_piles, pile_feature_dim]
            batch_source_mask: [B, num_from_piles] (bool tensor)
            batch_dest_mask: [B, num_to_piles] (bool tensor)
            batch_action: [B, 2] (각 행: [chosen_source, chosen_dest])
        반환:
            joint_logprob: [B, 1], value: [B, 1], joint_entropy: [B, 1]
        """
        B = batch_pile_features.size(0)
        source_logits, dest_logits, value, _ = self.forward(batch_pile_features)

        if batch_source_mask is not None:
            masked_source_logits = source_logits.masked_fill(~batch_source_mask, -1e9)
        else:
            masked_source_logits = source_logits
        if batch_dest_mask is not None:
            masked_dest_logits = dest_logits.masked_fill(~batch_dest_mask, -1e9)
        else:
            masked_dest_logits = dest_logits

        src_dist = torch.distributions.Categorical(logits=masked_source_logits)
        dst_dist = torch.distributions.Categorical(logits=masked_dest_logits)

        chosen_source = batch_action[:, 0]
        chosen_dest = batch_action[:, 1]
        src_logprob = src_dist.log_prob(chosen_source)
        dst_logprob = dst_dist.log_prob(chosen_dest)
        joint_logprob = src_logprob + dst_logprob
        joint_entropy = src_dist.entropy() + dst_dist.entropy()

        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)
