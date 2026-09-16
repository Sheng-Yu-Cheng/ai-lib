import torch
import torch.nn as nn
from benchmarks.gymnasium.continuous_mountain_car import ContinuousMountainCar
from methods.reinforcement_learning.actor_critic.network import ActorCriticNetwork

lr = 1e-4
n_episodes = 1000
gamma = 0.99

env = ContinuousMountainCar(verbose = True)

model = ActorCriticNetwork(
    state_dim = env.state_dim, 
    hidden_dims = (64, 64, 64), 
    action_dim = env.action_dim, 
    activation = nn.ReLU, 
)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = lr
)

for episode in range(n_episodes):
    state, info = env.reset()
    episode_reward = 0
    step = 0
    done = False
    while not done:
        mean, std, value_now = model.forward(torch.as_tensor(state, dtype = torch.float32))
        action_distribution = torch.distributions.Normal(mean, std)
        action = action_distribution.sample()
        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
        done = terminated or truncated
        
        with torch.no_grad():
            _, _, value_next = model.forward(torch.as_tensor(next_state, dtype = torch.float32))
        advantage = reward + gamma * value_next - value_now
        actor_loss = - advantage.detach() * action_distribution.log_prob(action).sum()
        critic_loss = advantage.pow(2).mean()
        
        loss = actor_loss + critic_loss * 0.5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        episode_reward += reward
        step += 1
    
    if episode % 100 == 0:
        print(f"Episode {episode} | episode reward = {episode_reward}")