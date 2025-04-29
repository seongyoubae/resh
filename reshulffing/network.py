import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#####################################
# 모델에 필요한 상수들
#####################################
MAX_SOURCE = 30
MAX_DEST = 30

def init_weights(model, init_std):
    """ 가중치 초기화 함수 """
    if model is None: return
    modules_to_init = model.modules() if not isinstance(model, nn.Sequential) else model
    for m in modules_to_init:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def build_mlp(input_dim, output_dim, hidden_dim, num_layers, activation=nn.GELU, use_dropout=False, use_layernorm=False):
    """ MLP 빌더 함수 (잔차 연결 미포함) """
    if num_layers < 1: return nn.Identity()
    if num_layers == 1: return nn.Linear(input_dim, output_dim)
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    if use_dropout: layers.append(nn.Dropout(p=0.1))
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if use_dropout: layers.append(nn.Dropout(p=0.1))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

# --- 최종 추천 모델 ---
class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_actor_layers,
                 num_critic_layers,
                 actor_init_std,
                 critic_init_std,
                 max_stack,
                 # 모델 내부 하이퍼파라미터
                 num_heads,
                 use_dropout_actor=False,
                 use_dropout_critic=False,
                 # 파일 개수는 위 상수 사용
                 num_from_piles=MAX_SOURCE,
                 num_to_piles=MAX_DEST,
                 ):
        """
        추천 구조: 안정화 설정과 함께 사용 권장.
        - Pre-LN (attn_norm) 사용
        - 소스 선택: [개별+글로벌] -> MLP (Actor 2층)
        - 목적지 선택: Attention (개선된 K/V 사용)
        - Critic: Global -> MLP (Critic 3층)
        """
        super().__init__()
        self.max_stack = max_stack
        self.pile_feature_dim = max_stack + 1
        self.num_from_piles = num_from_piles
        self.num_to_piles = num_to_piles
        self.total_piles = num_from_piles + num_to_piles
        self.embed_dim = embed_dim

        # --- 입력 인코더 ---
        self.pile_encoder = nn.Sequential(
            nn.Linear(self.pile_feature_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        init_weights(self.pile_encoder, 1.0)

        # --- Self-Attention (Pre-LN 포함) ---
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # --- Actor: 소스 선택 (개별 + 글로벌 -> MLP) ---
        self.source_logit_calculator = build_mlp(
            input_dim=embed_dim * 2, output_dim=1, hidden_dim=embed_dim,
            num_layers=num_actor_layers,
            activation=nn.GELU,
            use_dropout=use_dropout_actor, use_layernorm=True
        )
        init_weights(self.source_logit_calculator, actor_init_std)

        # --- Actor: 목적지 선택 (Attention, 개선된 K/V) ---
        self.dest_embeddings = nn.Parameter(torch.randn(self.num_to_piles, embed_dim))
        self.dest_kv_combiner = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        init_weights(self.dest_kv_combiner, actor_init_std)
        self.dest_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dest_fc = nn.Linear(embed_dim, self.num_to_piles) # 최종 로짓 Linear
        init_weights(self.dest_fc, actor_init_std)

        # --- Critic (글로벌 -> MLP) ---
        self.critic_net = build_mlp(
             input_dim=embed_dim, output_dim=1, hidden_dim=embed_dim,
             num_layers=num_critic_layers,
             activation=nn.GELU,
             use_dropout=use_dropout_critic, use_layernorm=True
        )
        init_weights(self.critic_net, critic_init_std)


    def forward(self, pile_features, debug=False):
        B = pile_features.size(0)

        # 1. 입력 인코딩
        pile_emb = self.pile_encoder(pile_features)

        # 2. Self-Attention (Pre-LN)
        attn_input = self.attn_norm(pile_emb)
        attn_output, attn_weights = self.self_attention(attn_input, attn_input, attn_input)

        # 3. Global Representation 계산 (단순 평균 풀링)
        global_repr = attn_output.mean(dim=1)

        # --- Actor ---
        # 4. 소스 로짓
        source_reprs = attn_output[:, :self.num_from_piles, :]
        global_expanded_src = global_repr.unsqueeze(1).expand(-1, self.num_from_piles, -1)
        combined_source_input = torch.cat((source_reprs, global_expanded_src), dim=-1)
        source_logits = self.source_logit_calculator(combined_source_input).squeeze(-1)

        # 5. 목적지 로짓
        dest_reprs = attn_output[:, self.num_from_piles:, :]
        dest_emb_expanded = self.dest_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined_dest_input = torch.cat((dest_emb_expanded, dest_reprs), dim=-1)
        dest_kv = self.dest_kv_combiner(combined_dest_input) # 개선된 Key/Value
        query = global_repr.unsqueeze(1) # Global Query
        attn_output_dest, _ = self.dest_attention(query=query, key=dest_kv, value=dest_kv)
        dest_logits = self.dest_fc(attn_output_dest.squeeze(1))

        # --- Critic ---
        # 6. 가치 예측
        value = self.critic_net(global_repr)

        if debug: pass

        return source_logits, dest_logits, value, global_repr

    @torch.no_grad()
    def act_batch(self, pile_features, source_masks=None, dest_masks=None, greedy=False, debug=False):
        self.eval()
        B = pile_features.size(0)
        source_logits, dest_logits, value, global_repr = self.forward(pile_features, debug=debug)
        if source_masks is not None:
            masked_source_logits = source_logits.masked_fill(~source_masks, -1e9)
        else: masked_source_logits = source_logits
        if dest_masks is not None:
            masked_dest_logits = dest_logits.masked_fill(~dest_masks, -1e9)
        else: masked_dest_logits = dest_logits
        source_policy = F.softmax(masked_source_logits, dim=-1)
        dest_policy = F.softmax(masked_dest_logits, dim=-1)
        if greedy:
            selected_source = source_policy.argmax(dim=-1)
            selected_dest = dest_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(source_policy, 1).squeeze(-1)
            selected_dest = torch.multinomial(dest_policy, 1).squeeze(-1)
        chosen_src_logprob = torch.log(source_policy.gather(1, selected_source.unsqueeze(1)) + 1e-10).squeeze(1)
        chosen_dest_logprob = torch.log(dest_policy.gather(1, selected_dest.unsqueeze(1)) + 1e-10).squeeze(1)
        joint_logprob = chosen_src_logprob + chosen_dest_logprob
        actions = torch.stack([selected_source, selected_dest], dim=-1)
        return actions, joint_logprob, value.squeeze(-1), global_repr

    def evaluate(self, batch_pile_features, batch_source_mask, batch_dest_mask, batch_action):
        B = batch_pile_features.size(0)
        source_logits, dest_logits, value, _ = self.forward(batch_pile_features)
        if batch_source_mask is not None:
            masked_source_logits = source_logits.masked_fill(~batch_source_mask, -1e9)
        else: masked_source_logits = source_logits
        if batch_dest_mask is not None:
            masked_dest_logits = dest_logits.masked_fill(~batch_dest_mask, -1e9)
        else: masked_dest_logits = dest_logits
        src_dist = torch.distributions.Categorical(logits=masked_source_logits)
        dst_dist = torch.distributions.Categorical(logits=masked_dest_logits)
        chosen_source = batch_action[:, 0]
        chosen_dest = batch_action[:, 1]
        src_logprob = src_dist.log_prob(chosen_source)
        dst_logprob = dst_dist.log_prob(chosen_dest)
        joint_logprob = src_logprob + dst_logprob
        joint_entropy = src_dist.entropy() + dst_dist.entropy()
        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)