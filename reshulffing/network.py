import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import false

#####################################
# 최대 Pile 개수 설정
#####################################
MAX_SOURCE = 30
MAX_DEST   = 30

#####################################
# 입력 차원 고정
#####################################
PAD_INPUT_DIM = (MAX_SOURCE + MAX_DEST) * 3  # 180
TOTAL_INPUT_DIM = PAD_INPUT_DIM * 2            # 360

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



def init_weights(model, init_std):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 target_entropy=1.0,
                 use_temperature=false,
                 fixed_temperature=1.0,
                 num_actor_layers=3,   # cfg에서 불러온 값
                 num_critic_layers=3,  # cfg에서 불러온 값
                 actor_init_std=0.01,  # cfg.py에서 불러온 값
                 critic_init_std=0.5   # cfg.py에서 불러온 값
                 ):
        super(SteelPlateConditionalMLPModel, self).__init__()
        self.max_source = MAX_SOURCE
        self.max_dest = MAX_DEST
        self.target_entropy = target_entropy
        self.use_temperature = use_temperature

        if self.use_temperature:
            self.temperature_param = nn.Parameter(torch.tensor(0.0))
        else:
            self.fixed_temperature = fixed_temperature

        # State와 Mask 인코더 (입력 차원: PAD_INPUT_DIM -> embed_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(PAD_INPUT_DIM, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(PAD_INPUT_DIM, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 각 인코더 후 별도의 path를 통해 gradient 흐름을 분리
        self.state_path = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.mask_path = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        # 두 path의 결과를 합산한 후 최종 임베딩 계산 (입력 차원: embed_dim)
        self.combined_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # ----- Actor Head -----
        self.source_head = build_mlp(embed_dim, self.max_source, embed_dim, num_actor_layers, nn.GELU, False, use_layernorm=False)

        # cond_dest_head는 임베딩과 one-hot source를 연결하는데, 여기서 one-hot source의 차원은 max_source
        self.cond_dest_head = build_mlp(embed_dim + self.max_source, self.max_dest, embed_dim, num_actor_layers, nn.GELU, False, use_layernorm=True)

        init_weights(self.source_head, actor_init_std)
        init_weights(self.cond_dest_head, actor_init_std)

        # ----- Critic Head -----
        # self.critic_net = build_mlp(embed_dim, 1, embed_dim, num_critic_layers, nn.GELU, use_dropout=False)
        # self.critic_net = build_mlp(embed_dim + self.max_source, 1, embed_dim, num_critic_layers, nn.GELU,
        #                             use_dropout=False)
        self.critic_net = build_mlp(embed_dim + self.max_source, 1, embed_dim, num_critic_layers, nn.GELU,
                                    use_dropout=False, use_layernorm=False)

        # 모든 layer 초기화
        init_weights(self.critic_net, critic_init_std)

        # 마지막 Linear layer만 작게 초기화 (gain=0.01)
        # for m in self.critic_net.modules():
        #     if isinstance(m, nn.Linear) and m.out_features == 1:
        #         nn.init.orthogonal_(m.weight, gain=1.0)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def _get_temperature(self):
        return self.temperature_param.exp() if self.use_temperature else self.fixed_temperature

    def forward(self, state, mask, selected_source=None, debug=False):
        if debug:
            print("=== Debug: Forward Pass ===")
            print("Original state shape:", state.shape)
            print("Original mask shape:", mask.shape)
            print("State stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
                state.min().item(), state.max().item(), state.mean().item(), state.std().item()))
            print("Mask unique values:", mask.unique())

        # 1. 패딩 및 마스크 적용
        state_pad, mask_pad = pad_input_state_and_mask(state, mask)
        if debug:
            print("After padding, state shape:", state_pad.shape)
            print("After padding, mask shape:", mask_pad.shape)
            print("Padded state stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
                state_pad.min().item(), state_pad.max().item(), state_pad.mean().item(), state_pad.std().item()))
            print("Padded mask unique values:", mask_pad.unique())

        mask_pad = mask_pad.float()
        state_pad = state_pad * mask_pad  # 패딩된 부분 무시
        if debug:
            valid_values = state_pad[mask_pad.bool()]
            if valid_values.numel() > 0:
                print(
                    "After applying mask (valid only), state stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
                        valid_values.min().item(), valid_values.max().item(), valid_values.mean().item(),
                        valid_values.std().item()))
            else:
                print("After applying mask (valid only), no valid values found")

        # 2. 인코더 단계
        state_emb = self.state_encoder(state_pad)
        mask_emb = self.mask_encoder(mask_pad)
        if debug:
            print("State encoder output - mean: {:.4f}, std: {:.4f}".format(
                state_emb.mean().item(), state_emb.std().item()))
            print("Mask encoder output - mean: {:.4f}, std: {:.4f}".format(
                mask_emb.mean().item(), mask_emb.std().item()))

        # 3. 각 path 및 합산
        state_out = self.state_path(state_emb)
        mask_out = self.mask_path(mask_emb)
        if debug:
            print("State path output - mean: {:.4f}, std: {:.4f}".format(
                state_out.mean().item(), state_out.std().item()))
            print("Mask path output - mean: {:.4f}, std: {:.4f}".format(
                mask_out.mean().item(), mask_out.std().item()))

        combined = state_out + mask_out
        if debug:
            print("Combined (state_out + mask_out) - mean: {:.4f}, std: {:.4f}".format(
                combined.mean().item(), combined.std().item()))

        emb = self.combined_fc(combined)
        if debug:
            print("Final embedding (after combined_fc) - mean: {:.4f}, std: {:.4f}".format(
                emb.mean().item(), emb.std().item()))

        # 4. Critic 계산 - selected_source를 고려한 critic input 생성
        if selected_source is not None:
            source_onehot = F.one_hot(selected_source, num_classes=self.max_source).float()
            if debug:
                print("Selected source one-hot shape:", source_onehot.shape)
        else:
            source_onehot = torch.zeros((emb.shape[0], self.max_source), device=emb.device)
            if debug:
                print("No selected_source provided. Using zero vector instead")

        critic_input = torch.cat([emb, source_onehot], dim=-1)
        if debug:
            print("Critic input shape:", critic_input.shape)
            print("Critic network input (embedding + onehot) - mean: {:.4f}, std: {:.4f}".format(
                critic_input.mean().item(), critic_input.std().item()))

        x = critic_input
        for idx, layer in enumerate(self.critic_net):
            x = layer(x)
            if debug:
                print(
                    f"Critic layer {idx} ({layer.__class__.__name__}) output - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # # 여기서 마지막 Linear 레이어 (예: 인덱스 2)의 weight를 확인합니다.
        # w = self.critic_net[2].weight
        # print(
        #     f"[CHECK] critic_layer2 weight - max: {w.max().item():.4f}, min: {w.min().item():.4f}, mean: {w.mean().item():.4f}, std: {w.std().item():.4f}")
        # print(f"[CHECK] Any NaN in critic_layer2 weight? {torch.isnan(w).any().item()}")
        # print(f"[CHECK] Any INF in critic_layer2 weight? {torch.isinf(w).any().item()}")

        value = x  # 최종 critic output
        if debug:
            print("Final critic output (value) - shape:", value.shape, "values:", value.squeeze(-1))

        # 4. Critic 계산
        # x = emb
        # if debug:
        #     print("Critic network input (embedding) - mean: {:.4f}, std: {:.4f}".format(
        #         x.mean().item(), x.std().item()))
        # for idx, layer in enumerate(self.critic_net):
        #     x = layer(x)
        #     if debug:
        #         print(
        #             f"Critic layer {idx} ({layer.__class__.__name__}) output - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        # value = x
        # if debug:
        #     print("Final critic output (value) - shape:", value.shape, "values:", value.squeeze(-1))

        # 5. Actor 계산
        source_logits = self.source_head(emb)
        temperature = self._get_temperature()
        if self.use_temperature:
            source_policy = F.softmax(source_logits / temperature, dim=-1)
        else:
            source_policy = F.softmax(source_logits, dim=-1)
        if debug:
            print("Source logits - mean: {:.4f}, std: {:.4f}".format(
                source_logits.mean().item(), source_logits.std().item()))
            print("Temperature: {:.4f}".format(temperature.item() if self.use_temperature else temperature))
            print("Source policy - mean: {:.4f}, std: {:.4f}".format(
                source_policy.mean().item(), source_policy.std().item()))

        if selected_source is None:
            if debug:
                print("Critic output (value) without selected_source:", value.squeeze(-1))
            dest_policy = None
            return source_policy, dest_policy, value, emb

        # 6. Conditional Dest 계산
        source_onehot = F.one_hot(selected_source, num_classes=self.max_source).float()
        cond_input = torch.cat([emb, source_onehot], dim=-1)
        if debug:
            print("Conditional input (embedding + onehot) - mean: {:.4f}, std: {:.4f}".format(
                cond_input.mean().item(), cond_input.std().item()))
        dest_logits = self.cond_dest_head(cond_input)
        if self.use_temperature:
            dest_policy = F.softmax(dest_logits / temperature, dim=-1)
        else:
            dest_policy = F.softmax(dest_logits, dim=-1)
        if debug:
            print("Destination logits - mean: {:.4f}, std: {:.4f}".format(
                dest_logits.mean().item(), dest_logits.std().item()))
            print("Destination policy - mean: {:.4f}, std: {:.4f}".format(
                dest_policy.mean().item(), dest_policy.std().item()))
            print("Critic output (value) with selected_source:", value.squeeze(-1))

        return source_policy, dest_policy, value, emb

    @torch.no_grad()
    def act_batch(self, states, masks, source_masks=None, dest_masks=None, greedy=False, debug=False):
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
        actions = torch.stack([selected_source, selected_dest], dim=-1)

        self.last_source_probs = masked_src_policy.detach().clone()
        self.last_dest_probs   = masked_dest_policy.detach().clone()

        return actions, joint_logprob, value.squeeze(-1), None

    def evaluate(self, batch_state, batch_mask, batch_action, batch_source_mask, batch_dest_mask):
        B = batch_state.size(0)

        # --- FORWARD PASS ---
        chosen_source = batch_action[:, 0]
        source_policy, _, value, emb = self.forward(batch_state, batch_mask, selected_source=chosen_source)

        # source_policy, _, value, emb = self.forward(batch_state, batch_mask, selected_source=None)

        # # --- DEBUG: embedding과 value 확인 ---
        # print(f"[DEBUG][evaluate] emb mean: {emb.mean().item():.4f}, std: {emb.std().item():.4f}")
        # print(f"[DEBUG][evaluate] value mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
        # print(f"[DEBUG][evaluate] value: {value.view(-1)}")

        # --- DEBUG: critic_net 내부 layer별 출력 ---
        # x = emb.clone()
        # for idx, layer in enumerate(self.critic_net):
        #     x = layer(x)
        #     print(
        #         f"[DEBUG][critic_net] Layer {idx}: {layer.__class__.__name__} -> mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # --- SOURCE POLICY ---
        source_log = torch.log(source_policy + 1e-10)
        source_log = source_log.masked_fill(~batch_source_mask, float('-inf'))
        source_probs = F.softmax(source_log, dim=-1)
        src_dist = torch.distributions.Categorical(source_probs)

        chosen_source = batch_action[:, 0]
        src_logprob = src_dist.log_prob(chosen_source)

        # --- DESTINATION POLICY ---
        _, dest_policy, _, _ = self.forward(batch_state, batch_mask, selected_source=chosen_source)
        dest_log = torch.log(dest_policy + 1e-10)
        dest_log = dest_log.masked_fill(~batch_dest_mask, float('-inf'))
        dest_probs = F.softmax(dest_log, dim=-1)
        dst_dist = torch.distributions.Categorical(dest_probs)

        chosen_dest = batch_action[:, 1]
        dst_logprob = dst_dist.log_prob(chosen_dest)

        # --- JOINT POLICY & ENTROPY ---
        joint_logprob = src_logprob + dst_logprob
        joint_entropy = src_dist.entropy() + dst_dist.entropy()

        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)

    def update_target_critic(self, tau=0.005):
        for param, tparam in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tparam.data.copy_(tau * param.data + (1.0 - tau) * tparam.data)
