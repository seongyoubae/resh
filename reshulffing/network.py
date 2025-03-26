import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################
# 최대 Pile 개수 설정
#####################################
MAX_SOURCE = 30
MAX_DEST   = 30

#####################################
# 입력 차원 고정
#####################################
PAD_INPUT_DIM = (MAX_SOURCE + MAX_DEST) * 3  # 30+30 * 3 = 180
TOTAL_INPUT_DIM = PAD_INPUT_DIM * 2            # state와 mask를 concat하면 180*2 = 360

def pad_input_state_and_mask(state: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    state: (B, D) 형태, mask: (B, D) 형태.
    만약 D가 PAD_INPUT_DIM보다 작으면 부족분을 0으로 채우고,
    D가 PAD_INPUT_DIM보다 크면 앞부분부터 PAD_INPUT_DIM 크기로 자른다.
    """
    current_dim = state.size(-1)
    target_dim  = PAD_INPUT_DIM
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        padding_state = state.new_zeros(state.size(0), pad_size)
        padding_mask  = mask.new_zeros(mask.size(0), pad_size)
        state = torch.cat([state, padding_state], dim=-1)
        mask  = torch.cat([mask, padding_mask], dim=-1)
    elif current_dim > target_dim:
        state = state[:, :target_dim]
        mask  = mask[:, :target_dim]
    return state, mask

class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 target_entropy=1.0,
                 use_temperature=True,
                 fixed_temperature=1.0):
        super(SteelPlateConditionalMLPModel, self).__init__()
        self.max_source = MAX_SOURCE
        self.max_dest = MAX_DEST
        self.target_entropy = target_entropy
        self.use_temperature = use_temperature

        if self.use_temperature:
            self.temperature_param = nn.Parameter(torch.tensor(0.0))
        else:
            self.fixed_temperature = fixed_temperature

        # state와 mask를 각각 처리하는 인코더 구성
        self.state_encoder = nn.Sequential(
            nn.Linear(PAD_INPUT_DIM, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(PAD_INPUT_DIM, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 두 인코더의 결과를 결합하여 최종 임베딩 생성
        self.combined_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU()
        )

        # ----- Actor Head -----
        self.source_head = nn.Linear(embed_dim, self.max_source)
        self.cond_dest_head = nn.Linear(embed_dim + self.max_source, self.max_dest)

        # ----- Critic Head -----
        # Critic은 원본 state와 mask를 단순 concat한 입력(TOTAL_INPUT_DIM)을 사용
        self.critic_net = nn.Sequential(
            nn.Linear(TOTAL_INPUT_DIM, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 1)
        )
        self.target_critic_net = copy.deepcopy(self.critic_net)

    def _get_temperature(self):
        return self.temperature_param.exp() if self.use_temperature else self.fixed_temperature

    def forward(self, state, mask, selected_source=None):
        # state와 mask를 PAD_INPUT_DIM 크기로 맞춤
        state_pad, mask_pad = pad_input_state_and_mask(state, mask)  # 둘 다 (B, PAD_INPUT_DIM)
        # mask_pad를 Float 타입으로 변환 (예: 0.0, 1.0)
        mask_pad = mask_pad.float()

        # 각각 별도 인코더 처리
        state_emb = self.state_encoder(state_pad)  # (B, embed_dim)
        mask_emb = self.mask_encoder(mask_pad)  # (B, embed_dim)

        # 두 인코딩 결과를 결합하여 최종 임베딩 생성
        combined = torch.cat([state_emb, mask_emb], dim=-1)  # (B, embed_dim*2)
        emb = self.combined_fc(combined)  # (B, embed_dim)

        # Actor: source 정책 생성
        source_logits = self.source_head(emb)  # (B, max_source)
        temperature = self._get_temperature()
        source_policy = F.softmax(source_logits / temperature, dim=-1) if self.use_temperature else F.softmax(
            source_logits, dim=-1)

        # Critic: 원본 state와 mask concat한 입력 사용
        x = torch.cat([state_pad, mask_pad], dim=-1)  # (B, TOTAL_INPUT_DIM)
        value = self.critic_net(x)  # (B, 1)

        if selected_source is None:
            return source_policy, None, value, emb

        # Conditional Dest 처리
        source_onehot = F.one_hot(selected_source, num_classes=self.max_source).float()
        cond_input = torch.cat([emb, source_onehot], dim=-1)  # (B, embed_dim + max_source)
        dest_logits = self.cond_dest_head(cond_input)
        dest_policy = F.softmax(dest_logits / temperature, dim=-1) if self.use_temperature else F.softmax(dest_logits,
                                                                                                          dim=-1)
        return source_policy, dest_policy, value, emb

    @torch.no_grad()
    def act_batch(self, states, masks, source_masks=None, dest_masks=None, greedy=False, debug=False):
        """
        states: (B, D) 상태 벡터
        masks: (B, D) 해당 상태의 마스크 벡터
        source_masks, dest_masks: (B, max_source), (B, max_dest) (옵션)
        """
        self.eval()
        B = states.size(0)
        source_policy, _, value, emb = self.forward(states, masks, selected_source=None)
        if source_masks is None:
            source_masks = torch.ones(B, self.max_source, dtype=torch.bool, device=states.device)
        src_logits = torch.log(source_policy + 1e-10)
        src_logits = src_logits.masked_fill(~source_masks, float('-inf'))
        masked_src_policy = F.softmax(src_logits, dim=-1)

        if greedy:
            selected_source = masked_src_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(masked_src_policy, 1).squeeze(1)
        chosen_src_logprob = torch.gather(torch.log(masked_src_policy + 1e-10),
                                           1,
                                           selected_source.unsqueeze(1)).squeeze(1)

        # Dest 선택
        _, dest_policy, _, _ = self.forward(states, masks, selected_source=selected_source)
        if dest_masks is None:
            dest_masks = torch.ones(B, self.max_dest, dtype=torch.bool, device=states.device)
        dst_logits = torch.log(dest_policy + 1e-10)
        dst_logits = dst_logits.masked_fill(~dest_masks, float('-inf'))
        masked_dest_policy = F.softmax(dst_logits, dim=-1)

        if greedy:
            selected_dest = masked_dest_policy.argmax(dim=-1)
        else:
            selected_dest = torch.multinomial(masked_dest_policy, 1).squeeze(1)
        chosen_dest_logprob = torch.gather(torch.log(masked_dest_policy + 1e-10),
                                            1,
                                            selected_dest.unsqueeze(1)).squeeze(1)

        joint_logprob = chosen_src_logprob + chosen_dest_logprob
        actions = torch.stack([selected_source, selected_dest], dim=-1)  # (B, 2)

        self.last_source_probs = masked_src_policy.detach().clone()
        self.last_dest_probs   = masked_dest_policy.detach().clone()

        return actions, joint_logprob, value.squeeze(-1), None

    def evaluate(self, batch_state, batch_mask, batch_action, batch_source_mask, batch_dest_mask):
        """
        PPO 업데이트시 ratio, entropy 등 계산용.
        batch_state: (B, D), batch_mask: (B, D)
        batch_action: (B, 2)
        batch_source_mask, batch_dest_mask: (B, max_source), (B, max_dest)
        """
        B = batch_state.size(0)
        source_policy, _, value, _ = self.forward(batch_state, batch_mask, selected_source=None)
        source_log = torch.log(source_policy + 1e-10)
        source_log = source_log.masked_fill(~batch_source_mask, float('-inf'))
        source_probs = F.softmax(source_log, dim=-1)
        src_dist = torch.distributions.Categorical(source_probs)

        chosen_source = batch_action[:, 0]
        src_logprob = src_dist.log_prob(chosen_source)

        _, dest_policy, _, _ = self.forward(batch_state, batch_mask, selected_source=chosen_source)
        dest_log = torch.log(dest_policy + 1e-10)
        dest_log = dest_log.masked_fill(~batch_dest_mask, float('-inf'))
        dest_probs = F.softmax(dest_log, dim=-1)
        dst_dist = torch.distributions.Categorical(dest_probs)

        chosen_dest = batch_action[:, 1]
        dst_logprob = dst_dist.log_prob(chosen_dest)

        joint_logprob = src_logprob + dst_logprob
        joint_entropy = src_dist.entropy() + dst_dist.entropy()

        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)

    def update_target_critic(self, tau=0.005):
        for param, tparam in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tparam.data.copy_(tau * param.data + (1.0 - tau) * tparam.data)
