import argparse
import os

def get_cfg():
    parser = argparse.ArgumentParser(description="Steel Plate Selection RL Hyperparameters")

    # 학습 관련
    parser.add_argument("--n_episode", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--load_model", action="store_true", help="Whether to load a pre-trained model")
    parser.add_argument("--model_path", type=str, default="", help="Path to the pre-trained model file")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Dimension of node embeddings")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of multi-head attention heads")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="Number of HGT layers")
    parser.add_argument("--num_actor_layers", type=int, default=3, help="Number of layers in the actor network")
    parser.add_argument("--num_critic_layers", type=int, default=3, help="Number of layers in the critic network")
    parser.add_argument("--temp_lr", type=float, default=0.01, help="Learning rate for temperature parameter")
    parser.add_argument("--target_entropy", type=float, default=2.0, help="Target entropy for automatic temperature adjustment")
    parser.add_argument("--lr", type=float, default=0.0005, help="General learning rate (if not using separate actor/critic lr)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=0.003, help="Learning rate for critic network")
    parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay factor")
    parser.add_argument("--lr_step", type=int, default=2000, help="Step interval for learning rate decay")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter (lambda)")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--T_horizon", type=int, default=50, help="Number of steps to collect per update, step in episode")
    parser.add_argument("--episodes_per_epoch", type=int, default=64, help="Number of episodes to collect per epoch, mini batch size")
    parser.add_argument("--mini_batch_size", type=int, default=50, help="Fixed size of minibatches per PPO update, same with T_horizon")
    parser.add_argument("--K_epoch", type=int, default=5, help="Number of optimization epochs per update")
    parser.add_argument("--n_epoch", type=int, default=50000, help="Total number of epochs")
    parser.add_argument("--P_coeff", type=float, default=1.0, help="Coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="Coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="Coefficient for entropy loss")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=500, help="Save model every x episodes")
    parser.add_argument("--save_final_state_every", type=int, default=10000, help="Save final state every x episodes")
    parser.add_argument("--new_instance_every", type=int, default=30000, help="Generate new scenarios every x episodes")
    parser.add_argument("--value_clip_range", type=float, default=1.0, help="Value clipping range for critic update")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm for model parameters")

    # 파일 및 저장 관련
    parser.add_argument("--plates_data_path", type=str, default="output/reshuffle_plan.xlsx", help="Path to the plates data Excel file")
    parser.add_argument("--evaluation_plates_data_path", type=str, default="output/reshuffle_plan(for eval).xlsx", help="Path to the plates eval data Excel file")
    parser.add_argument("--log_file", type=str, default="training_log.csv", help="Path for training log CSV file")
    parser.add_argument("--save_model_dir", type=str, default="saved_models", help="Directory to save models")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output Excel files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--tensorboard_dir", type=str, default="./runs", help="Directory to save Tensorboard logs")

    # 환경 관련 파라미터
    parser.add_argument("--num_pile", type=int, default=4, help="Number of piles (files) in the environment/action space")
    parser.add_argument("--max_stack", type=int, default=30, help="Maximum stack height for each pile")
    parser.add_argument("--crane_penalty", type=float, default=0, help="Crane movement penalty")

    # 데이터 생성 관련
    parser.add_argument("--num_plates", type=int, default=150, help="Number of plates for schedule generation")
    parser.add_argument("--inbound_min", type=int, default=1, help="Minimum inbound value")
    parser.add_argument("--inbound_max", type=int, default=10, help="Maximum inbound value")
    parser.add_argument("--outbound_extra_min", type=int, default=1, help="Minimum extra for outbound relative to inbound")
    parser.add_argument("--outbound_extra_max", type=int, default=20, help="Maximum extra for outbound relative to inbound")
    parser.add_argument("--unitw_min", type=float, default=0.141, help="Minimum unit weight")
    parser.add_argument("--unitw_max", type=float, default=19.294, help="Maximum unit weight")

    # 재배치 계획 관련
    parser.add_argument("--n_from_piles_reshuffle", type=int, default=5, help="Number of source piles for reshuffle")
    parser.add_argument("--n_to_piles_reshuffle", type=int, default=5, help="Number of destination piles for reshuffle")
    parser.add_argument("--n_plates_reshuffle", type=int, default=10, help="Average number of plates to reshuffle per pile")
    parser.add_argument("--safety_margin", type=int, default=0, help="Safety margin for reshuffle plan")

    # 네트워크 초기화 관련
    parser.add_argument("--actor_init_std", type=float, default=0.01, help="Standard deviation for actor head initialization")
    parser.add_argument("--critic_init_std", type=float, default=1.0, help="Standard deviation for critic head initialization")
    parser.add_argument("--activation", type=str, default="ELU", help="Activation function to use (e.g., elu, relu)")

    # 타깃 critic 업데이트 관련 (추가)
    parser.add_argument("--update_target_interval", type=int, default=10, help="Interval (in minibatches) for updating target critic network")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update coefficient for target critic network")

    args = parser.parse_args()

    # Vessl Web UI에서 설정된 하이퍼파라미터로 덮어쓰기
    for key, value in vars(args).items():
        env_key = f"VESSEL_PARAM_{key}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if isinstance(value, bool):
                    setattr(args, key, env_value.lower() == "true")
                elif isinstance(value, int):
                    setattr(args, key, int(env_value))
                elif isinstance(value, float):
                    setattr(args, key, float(env_value))
                else:
                    setattr(args, key, env_value)
            except Exception as e:
                print(f"[Warning] Could not convert {env_key}='{env_value}' to type {type(value)}: {e}")

    return args