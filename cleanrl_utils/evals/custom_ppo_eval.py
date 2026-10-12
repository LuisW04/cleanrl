from typing import Callable

import gymnasium as gym
import torch
import numpy as np


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    SupportModel : torch.nn.Module,
    feature_mask : list[int],
    use_support_model : bool = False,
    support_model_path : str = "",
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name)])
    agent = Model(envs, use_support_model, feature_mask).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    if use_support_model:
        support_model = SupportModel().to(device)
        support_model.load_state_dict(torch.load(support_model_path, map_location=device))
        support_model.eval()

    obs, _ = envs.reset()
    
    
    last_actions = torch.zeros((envs.num_envs, 1)).to(device)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        obs = torch.Tensor(obs).to(device)

        
        with torch.no_grad():
            if use_support_model:
                prediction = support_model(obs, last_actions)[:, feature_mask]
                actions, _, _, _ = agent.get_action_and_value(obs, prediction)
                last_actions = actions
            else:
                actions, _, _, _ = agent.get_action_and_value(obs)
            
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from cleanrl.ppo_atari import Agent, MsPacmanModel, make_env

    model_path = "runs/ALE/MsPacman-v5__cuu_6_features__1__1789468246/model.pt"
    support_model_path = "support_models/model_20_files.pt"
    feature_mask = np.array(range(6))
    results = evaluate(
        model_path,
        make_env,
        "ALE/MsPacman-v5",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        SupportModel=MsPacmanModel,
        feature_mask=feature_mask,
        use_support_model=True,
        support_model_path=support_model_path,
        device="cpu",
        capture_video=True,
    )
    sum = np.sum(results)
    print("average: ", sum/ len(results))
