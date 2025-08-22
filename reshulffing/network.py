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
            # nn.init.orthogonal_(m.weight, gain=init_std) # Orthogonal 초기화
            nn.init.xavier_uniform_(m.weight, gain=init_std) # Xavier 초기화 (더 일반적일 수 있음)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                # nn.init.orthogonal_(m.in_proj_weight, gain=init_std)
                nn.init.xavier_uniform_(m.in_proj_weight, gain=init_std)
            if m.out_proj.weight is not None:
                # nn.init.orthogonal_(m.out_proj.weight, gain=init_std)
                nn.init.xavier_uniform_(m.out_proj.weight, gain=init_std)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
        elif isinstance(m, nn.LayerNorm):
             if m.elementwise_affine:
                 nn.init.ones_(m.weight)
                 nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    """
    MLP 내에서 사용될 잔차 블록 (Linear -> Norm? -> Activation -> Dropout? + Skip Connection)
    입력과 출력의 차원이 동일해야 함 (hidden_dim)
    """
    def __init__(self, hidden_dim, activation=nn.GELU, use_dropout=False, use_layernorm=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.use_dropout = use_dropout

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        if self.use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)
        self.activation = activation()
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1) # 드롭아웃 확률은 필요시 조정

    def forward(self, x):
        identity = x # 입력 값 저장 (Skip connection 용)

        out = self.fc(x)
        if self.use_layernorm:
            out = self.norm(out)
        out = self.activation(out)
        if self.use_dropout:
            out = self.dropout(out)

        # --- 잔차 연결 ---
        out = out + identity
        # --- ----------- ---

        return out

def build_mlp_with_residuals(input_dim, output_dim, hidden_dim, num_layers, activation=nn.GELU, use_dropout=False, use_layernorm=False):
    """ ResidualBlock을 사용하여 MLP를 구축하는 함수 """
    if num_layers < 1:
        return nn.Identity()
    if num_layers == 1:
        # 레이어가 하나면 Linear 레이어만 반환
        return nn.Linear(input_dim, output_dim)

    layers = []
    # === 입력 레이어: input_dim -> hidden_dim ===
    layers.append(nn.Linear(input_dim, hidden_dim))
    # 입력 레이어 뒤 Norm/Activation/Dropout (선택적 순서)
    if use_layernorm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    if use_dropout:
        layers.append(nn.Dropout(p=0.1))

    # === 중간 Residual 블록들: hidden_dim -> hidden_dim ===
    # 총 num_layers 중 입력층과 출력층을 제외한 (num_layers - 2)개의 ResidualBlock 추가
    for _ in range(num_layers - 2):
        layers.append(ResidualBlock(hidden_dim, activation, use_dropout, use_layernorm))

    # === 출력 레이어: hidden_dim -> output_dim ===
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)

class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_actor_layers,
                 num_critic_layers,
                 actor_init_std,
                 critic_init_std,
                 pile_feature_dim,
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
        - 소스 선택: [개별+글로벌] -> MLP (Actor MLP)
        - 목적지 선택: Attention (개선된 K/V 사용)
        - Critic: Attention Pooling -> MLP (Critic MLP) # <--- 변경됨
        """
        super().__init__()
        self.pile_feature_dim = pile_feature_dim
        self.num_from_piles = num_from_piles
        self.num_to_piles = num_to_piles
        self.total_piles = self.num_from_piles + self.num_to_piles
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

        init_weights(self.self_attention, 1.0)

        # --- Actor: 소스 선택 (개별 + 글로벌 -> MLP) ---
        self.source_logit_calculator = build_mlp_with_residuals(
            input_dim=embed_dim * 2, output_dim=1, hidden_dim=embed_dim,
            num_layers=num_actor_layers,
            activation=nn.GELU,
            use_dropout=use_dropout_actor, use_layernorm=True
        )
        init_weights(self.source_logit_calculator, actor_init_std)

        # --- Actor: 목적지 선택 (Attention, 개선된 K/V) ---
        self.dest_embeddings = nn.Parameter(torch.randn(self.num_to_piles, embed_dim))
        nn.init.xavier_uniform_(self.dest_embeddings) # 목적지 임베딩 초기화
        self.dest_kv_combiner = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        init_weights(self.dest_kv_combiner, actor_init_std)
        self.dest_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        init_weights(self.dest_attention, actor_init_std) # 목적지 어텐션 가중치 초기화
        self.dest_fc = nn.Linear(embed_dim, self.num_to_piles) # 최종 로짓 Linear
        init_weights(self.dest_fc, actor_init_std)

        # --- Critic (Attention Pooling -> MLP) --- # <--- 변경된 부분
        # 1. Critic용 Attention Pooling 레이어
        self.critic_query = nn.Parameter(torch.randn(1, 1, embed_dim)) # 학습 가능한 쿼리 벡터
        nn.init.xavier_uniform_(self.critic_query) # 쿼리 벡터 초기화
        # MultiheadAttention을 사용하여 풀링 구현 (num_heads=1 또는 num_heads 사용 가능)
        self.critic_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        init_weights(self.critic_attention, critic_init_std) # Critic 어텐션 가중치 초기화

        # 2. Critic MLP 네트워크
        self.critic_net = build_mlp_with_residuals(
             input_dim=embed_dim, # 어텐션 풀링 결과 차원은 embed_dim
             output_dim=1, hidden_dim=embed_dim,
             num_layers=num_critic_layers,
             activation=nn.GELU,
             use_dropout=use_dropout_critic, use_layernorm=True
        )
        init_weights(self.critic_net, critic_init_std)


    def forward(self, pile_features, debug=False):
        B = pile_features.size(0)

        # 1. 입력 인코딩
        pile_emb = self.pile_encoder(pile_features) # (B, total_piles, embed_dim)

        # 2. Self-Attention (Pre-LN)
        attn_input = self.attn_norm(pile_emb)
        # attn_output: (B, total_piles, embed_dim)
        attn_output, attn_weights = self.self_attention(attn_input, attn_input, attn_input)

        x_after_attn_residual = pile_emb + attn_output

        # 3. Actor용 Global Representation 계산 (단순 평균 풀링)
        # Critic은 아래 Attention Pooling을 사용하므로, Actor 로직 유지를 위해 별도 계산
        actor_global_repr = x_after_attn_residual.mean(dim=1) # (B, embed_dim)

        # --- Actor ---
        # 4. 소스 로짓 계산 (actor_global_repr 사용)
        source_reprs = x_after_attn_residual[:, :self.num_from_piles, :]
        global_expanded_src = actor_global_repr.unsqueeze(1).expand(-1, self.num_from_piles, -1)
        combined_source_input = torch.cat((source_reprs, global_expanded_src), dim=-1) # (B, num_from_piles, embed_dim * 2)
        source_logits = self.source_logit_calculator(combined_source_input).squeeze(-1) # (B, num_from_piles)

        # 5. 목적지 로짓 계산 (actor_global_repr 사용)
        dest_reprs = x_after_attn_residual[:, self.num_from_piles:, :]
        dest_emb_expanded = self.dest_embeddings.unsqueeze(0).expand(B, -1, -1) # (B, num_to_piles, embed_dim)
        combined_dest_input = torch.cat((dest_emb_expanded, dest_reprs), dim=-1) # (B, num_to_piles, embed_dim * 2)
        dest_kv = self.dest_kv_combiner(combined_dest_input) # 개선된 Key/Value (B, num_to_piles, embed_dim)
        # Global Query로 actor_global_repr 사용
        query_dest = actor_global_repr.unsqueeze(1) # (B, 1, embed_dim)
        attn_output_dest, _ = self.dest_attention(query=query_dest, key=dest_kv, value=dest_kv) # (B, 1, embed_dim)
        dest_logits = self.dest_fc(attn_output_dest.squeeze(1)) # (B, num_to_piles)

        # --- Critic --- # <--- 변경된 부분
        # 6. Critic 입력 계산 (Attention Pooling)
        # 학습 가능한 쿼리를 배치 크기에 맞게 확장
        critic_q = self.critic_query.expand(B, -1, -1)
        critic_pooled_output, critic_attn_weights = self.critic_attention(
            query=critic_q,
            key=x_after_attn_residual,
            value=x_after_attn_residual
        )
        critic_input = critic_pooled_output.squeeze(1)

        # 7. 가치 예측
        value = self.critic_net(critic_input) # (B, 1)

        if debug:
            # 디버깅 필요시 critic_attn_weights 등을 확인 가능
            print("Critic Attention Weights Shape:", critic_attn_weights.shape) # (B, 1, total_piles)

        # actor_global_repr 반환 (act_batch 등에서 필요시 사용 가능)
        return source_logits, dest_logits, value, actor_global_repr

    @torch.no_grad()
    def act_batch(self, pile_features, source_masks=None, dest_masks=None, greedy=False, debug=False):
        self.eval()
        B = pile_features.size(0)
        # forward 호출 시 actor_global_repr가 4번째 값으로 반환됨
        source_logits, dest_logits, value, actor_global_repr = self.forward(pile_features, debug=debug)

        if source_masks is not None:
            masked_source_logits = source_logits.masked_fill(~source_masks, -float('inf')) # -1e9 대신 -inf 사용 권장
        else: masked_source_logits = source_logits
        if dest_masks is not None:
            masked_dest_logits = dest_logits.masked_fill(~dest_masks, -float('inf')) # -1e9 대신 -inf 사용 권장
        else: masked_dest_logits = dest_logits

        source_policy = F.softmax(masked_source_logits, dim=-1)
        dest_policy = F.softmax(masked_dest_logits, dim=-1)

        if greedy:
            selected_source = source_policy.argmax(dim=-1)
            selected_dest = dest_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(source_policy, 1).squeeze(-1)
            # multinomial 입력값이 0만 있는 경우 방지 (매우 작은 값 추가)
            dest_policy = dest_policy + 1e-10
            dest_policy = dest_policy / dest_policy.sum(dim=-1, keepdim=True)
            selected_dest = torch.multinomial(dest_policy, 1).squeeze(-1)

        # 로그 확률 계산 시 작은 값(epsilon) 추가 안정성 확보
        chosen_src_logprob = torch.log(source_policy.gather(1, selected_source.unsqueeze(1)) + 1e-10).squeeze(1)
        chosen_dest_logprob = torch.log(dest_policy.gather(1, selected_dest.unsqueeze(1)) + 1e-10).squeeze(1)

        joint_logprob = chosen_src_logprob + chosen_dest_logprob
        actions = torch.stack([selected_source, selected_dest], dim=-1)

        # value는 (B, 1) 형태이므로 squeeze(-1) 필요
        # actor_global_repr는 그대로 반환 (필요시 사용)
        return actions, joint_logprob, value.squeeze(-1), actor_global_repr

    def evaluate(self, batch_pile_features, batch_source_mask, batch_dest_mask, batch_action):
        """
        PPO 업데이트 시 사용: 새로운 정책의 로그 확률, 상태 가치, 엔트로피를 계산합니다.
        안정성을 위해 NaN 검사 및 '유효 행동 없음' 엣지 케이스 처리 로직을 포함합니다.
        """
        self.train()  # PPO 업데이트는 학습 과정이므로 train 모드로 설정

        # 1. 모델 순전파: 로짓과 가치 예측
        source_logits, dest_logits, value, _ = self.forward(batch_pile_features)

        # 2. NaN 값 명시적 검사 (근본 원인 진단)
        # 모델 출력 자체에 NaN이 있다면, 이는 학습률 문제 등 심각한 수치적 불안정성이므로
        # 안전장치로 넘어가기보다 즉시 에러를 발생시켜 원인을 수정하도록 유도합니다.
        if torch.isnan(source_logits).any() or torch.isnan(dest_logits).any():
            raise ValueError(
                "NaN detected in model logits. This is a critical numerical instability issue. "
                "Consider lowering the learning rate, checking reward scaling, or implementing gradient clipping."
            )

        # 3. 유효하지 않은 행동에 마스크 적용
        masked_source_logits = source_logits.masked_fill(~batch_source_mask, -float('inf'))
        masked_dest_logits = dest_logits.masked_fill(~batch_dest_mask, -float('inf'))

        # 4. '유효 행동 없음' 엣지 케이스 처리 (안전장치)
        # 마스크로 인해 모든 로짓이 -inf가 되면 Categorical 분포 생성 시 NaN이 발생합니다.
        # 이 경우에만 해당하는 샘플을 찾아 안전하게 처리합니다.

        # 소스 로짓 처리
        all_source_invalid = torch.all(~batch_source_mask, dim=-1)
        if torch.any(all_source_invalid):
            # 문제가 되는 배치 샘플에 대해서만, 첫 번째 행동[0]에 0의 로짓을 부여합니다.
            # 이렇게 하면 해당 행동의 확률이 1인 유효한 분포가 만들어져 크래시를 방지합니다.
            # 이 조치는 학습에 거의 영향을 주지 않고 안정성만 높입니다.
            masked_source_logits[all_source_invalid, 0] = 0.0

        # 목적지 로짓 처리 (소스와 동일)
        all_dest_invalid = torch.all(~batch_dest_mask, dim=-1)
        if torch.any(all_dest_invalid):
            masked_dest_logits[all_dest_invalid, 0] = 0.0

        # 5. 확률 분포 생성 및 값 계산
        src_dist = torch.distributions.Categorical(logits=masked_source_logits)
        dst_dist = torch.distributions.Categorical(logits=masked_dest_logits)

        chosen_source = batch_action[:, 0]
        chosen_dest = batch_action[:, 1]

        src_logprob = src_dist.log_prob(chosen_source)
        dst_logprob = dst_dist.log_prob(chosen_dest)
        joint_logprob = src_logprob + dst_logprob

        joint_entropy = src_dist.entropy() + dst_dist.entropy()

        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)