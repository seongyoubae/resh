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

def pad_input_state(state: torch.Tensor) -> torch.Tensor:
    """
    state: (B, D) 형태
    -> 부족분 0 패딩, 초과분 잘라내서 (B, 180)으로 맞춤
    """
    current_dim = state.size(-1)
    target_dim  = PAD_INPUT_DIM
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        padding = state.new_zeros(state.size(0), pad_size)
        state = torch.cat([state, padding], dim=-1)
    elif current_dim > target_dim:
        state = state[:, :target_dim]
    return state

class SteelPlateConditionalMLPModel(nn.Module):
    """
    - Actor: source_head(embed_dim->max_source=30), dest_head(embed_dim+30->30)
    - Critic: critic_net(180->1)
      -> 모두 pad_input_state로 (B,180)을 맞춘 뒤 처리
    """
    def __init__(self,
                 embed_dim=128,
                 num_actor_layers=2,
                 num_critic_layers=2,
                 max_source=MAX_SOURCE,
                 max_dest=MAX_DEST,
                 target_entropy=1.0,
                 use_temperature=True,
                 fixed_temperature=1.0):
        super(SteelPlateConditionalMLPModel, self).__init__()
        self.max_source = max_source
        self.max_dest   = max_dest
        self.target_entropy = target_entropy
        self.use_temperature = use_temperature

        if self.use_temperature:
            self.temperature_param = nn.Parameter(torch.tensor(0.0))  # 학습할 값
        else:
            self.fixed_temperature = fixed_temperature

        # ----- Actor Shared Net -----
        actor_layers = []
        in_dim = PAD_INPUT_DIM  # 180
        for _ in range(num_actor_layers - 1):
            actor_layers.append(nn.Linear(in_dim, embed_dim))
            actor_layers.append(nn.ELU())
            in_dim = embed_dim
        actor_layers.append(nn.Linear(in_dim, embed_dim))
        actor_layers.append(nn.ELU())
        self.shared_net = nn.Sequential(*actor_layers)

        # ----- Source Head (embed_dim->max_source) -----
        self.source_head = nn.Linear(embed_dim, max_source)

        # ----- Conditional Dest Head (embed_dim+max_source->max_dest) -----
        self.cond_dest_head = nn.Linear(embed_dim + max_source, max_dest)

        # ----- Critic Net (180->1) -----
        critic_layers = []
        c_in_dim = PAD_INPUT_DIM
        for _ in range(num_critic_layers - 1):
            critic_layers.append(nn.Linear(c_in_dim, embed_dim))
            critic_layers.append(nn.ELU())
            c_in_dim = embed_dim
        critic_layers.append(nn.Linear(c_in_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)

        # Target Critic
        self.target_critic_net = copy.deepcopy(self.critic_net)

    def _get_temperature(self):
        if self.use_temperature:
            return self.temperature_param.exp()
        else:
            return self.fixed_temperature

    def forward(self, state, selected_source=None):
        """
        Actor 전방향:
          state: (B, ?) -> pad -> (B,180)
          shared_net -> (B, embed_dim)
          source_head -> (B, max_source)
          if selected_source is None -> return source_policy only
          else -> dest_policy까지 계산
        Critic는 별도 함수 없이 여기서도 value를 같이 반환
        """
        state_pad = pad_input_state(state)           # (B,180)
        emb = self.shared_net(state_pad)             # (B, embed_dim)
        source_logits = self.source_head(emb)        # (B, max_source)
        temperature = self._get_temperature()

        if self.use_temperature:
            source_policy = F.softmax(source_logits/temperature, dim=-1)
        else:
            source_policy = F.softmax(source_logits, dim=-1)

        value = self.critic_net(state_pad)           # (B,1)

        if selected_source is None:
            return source_policy, None, value, emb

        # Conditional Dest
        source_onehot = F.one_hot(selected_source, num_classes=self.max_source).float()
        cond_input = torch.cat([emb, source_onehot], dim=-1)  # (B, embed_dim+max_source)
        dest_logits = self.cond_dest_head(cond_input)
        if self.use_temperature:
            dest_policy = F.softmax(dest_logits / temperature, dim=-1)
        else:
            dest_policy = F.softmax(dest_logits, dim=-1)

        return source_policy, dest_policy, value, emb

    @torch.no_grad()
    def act_batch(self, states, source_masks=None, dest_masks=None, greedy=False, debug=False):
        """
        states: (B, ?)
        source_masks, dest_masks: (B, max_source) / (B, max_dest)
        """
        self.eval()
        B = states.size(0)

        # Source
        source_policy, _, value, emb = self.forward(states, selected_source=None)
        if source_masks is None:
            source_masks = torch.ones(B, self.max_source, dtype=torch.bool, device=states.device)
        src_logits = torch.log(source_policy + 1e-10)
        src_logits = src_logits.masked_fill(~source_masks, float('-inf'))
        masked_src_policy = F.softmax(src_logits, dim=-1)

        if greedy:
            selected_source = masked_src_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(masked_src_policy, 1).squeeze(1)
        chosen_src_logprob = torch.gather(
            torch.log(masked_src_policy+1e-10), 1, selected_source.unsqueeze(1)
        ).squeeze(1)

        # Dest
        _, dest_policy, _, _ = self.forward(states, selected_source=selected_source)
        if dest_masks is None:
            dest_masks = torch.ones(B, self.max_dest, dtype=torch.bool, device=states.device)
        dst_logits = torch.log(dest_policy + 1e-10)
        dst_logits = dst_logits.masked_fill(~dest_masks, float('-inf'))
        masked_dest_policy = F.softmax(dst_logits, dim=-1)

        if greedy:
            selected_dest = masked_dest_policy.argmax(dim=-1)
        else:
            selected_dest = torch.multinomial(masked_dest_policy, 1).squeeze(1)
        chosen_dest_logprob = torch.gather(
            torch.log(masked_dest_policy+1e-10), 1, selected_dest.unsqueeze(1)
        ).squeeze(1)

        joint_logprob = chosen_src_logprob + chosen_dest_logprob
        actions = torch.stack([selected_source, selected_dest], dim=-1)  # (B,2)

        self.last_source_probs = masked_src_policy.detach().clone()
        self.last_dest_probs   = masked_dest_policy.detach().clone()

        return actions, joint_logprob, value.squeeze(-1), None

    def evaluate(self, batch_state, batch_action, batch_source_mask, batch_dest_mask):
        """
        PPO 업데이트시 ratio, entropy 등 계산용
        batch_action: (B,2)
        batch_source_mask, batch_dest_mask: (B, max_source), (B, max_dest)
        """
        B = batch_state.size(0)
        source_policy, _, value, _ = self.forward(batch_state, selected_source=None)
        source_log = torch.log(source_policy + 1e-10)
        source_log = source_log.masked_fill(~batch_source_mask, float('-inf'))
        source_probs = F.softmax(source_log, dim=-1)
        src_dist = torch.distributions.Categorical(source_probs)

        chosen_source = batch_action[:, 0]
        src_logprob = src_dist.log_prob(chosen_source)

        _, dest_policy, _, _ = self.forward(batch_state, selected_source=chosen_source)
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
            tparam.data.copy_(tau * param.data + (1.0 - tau)*tparam.data)
