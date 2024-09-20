import sys
sys.path.append('/home/dukaifeng/my_project/panda_gym_cam-master/panda_gym')
import panda_gym
import gymnasium as gym
import time
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.functional import F
from torch.distributions.normal import Normal
from tqdm import tqdm
import torch.optim as optim



class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
        else:
            self.conv3 = None

    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class Agent(nn.Module):

    def __init__(self,device='cpu'):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
        b2 = nn.Sequential(Residual(64,64), Residual(64, 64))
        b3 = nn.Sequential(Residual(64,128,2), Residual(128, 128))
        b4 = nn.Sequential(Residual(128,256,2), Residual(256, 256))
        b5 = nn.Sequential(Residual(256,512,2), Residual(512, 512))
        backbone = nn.Sequential(b1,b2,b3,b4,b5) #(1, 512, 3, 3)
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.critic = nn.Sequential(
                                    backbone,
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(512, 64), nn.ReLU(),
                                    nn.Linear(64,1))

        self.actor_mean = nn.Sequential(backbone,
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(512, 128, nn.ReLU()),
                                        nn.Linear(128,3), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.zeros(1,3))

        self.action_low = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.action_high = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
        self.action_range = self.action_high - self.action_low

        self.apply(self._orthogonal_init)


    def _orthogonal_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, bias_const)



    def get_value(self, obs):
        obs = (obs/255.0).permute(0,3,1,2).to(DEVICE)
        return self.critic(obs).reshape((1,-1))


    def get_action_and_value(self, obs, action=None):

        obs = (obs/255.0).permute(0,3,1,2).to(DEVICE)
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        # if torch.isnan(action_mean).any():
        #     print('----------------errer---------------')
            
        #     time.sleep(100000)
        # if  torch.isnan(action_std).any():
        #     print('----------------errer---------------')
        #     time.sleep(100000)            

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = (((F.tanh(action)+1) / 2) * self.action_range) + self.action_low
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(obs).reshape((1,-1))



if __name__ == "__main__":


    DEVICE = torch.device('cuda')
    NUM_STEPS = 1200   
    NUM_ENVS = 1
    NUM_UPDATES = 200
    mini_batch_size = 25
    mini_batch_num = NUM_STEPS/mini_batch_size
    update_epochs = 10
    gamma = 0.99
    gae_lamda = 0.95
    clip_epsilon = 0.2
    optimizer_lr = 0.0003



    # env = gym.make('CarRacing-v2')
    env_id = 'CarRacing-v2' 
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for i in range(NUM_ENVS)])
    agent = Agent(DEVICE).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(),lr=optimizer_lr,eps=1e-8)


    obs = torch.zeros((NUM_STEPS,NUM_ENVS) + (envs.single_observation_space.shape)).to(DEVICE)
    terminateds = torch.zeros((NUM_STEPS,NUM_ENVS)).to(DEVICE)
    actions = torch.zeros((NUM_STEPS,NUM_ENVS) + envs.single_action_space.shape).to(DEVICE)
    logprobs = torch.zeros((NUM_STEPS,NUM_ENVS)).to(DEVICE)               
    rewards = torch.zeros((NUM_STEPS,NUM_ENVS)).to(DEVICE)
    values = torch.zeros((NUM_STEPS,NUM_ENVS)).to(DEVICE)
    advantages = torch.zeros((NUM_STEPS,NUM_ENVS)).to(DEVICE)


    global_step = 0
    observation,_ = envs.reset()


    observation = torch.from_numpy(observation).to(DEVICE)
    terminated = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    counter = 0

    for update in range(1,NUM_UPDATES + 1):

        frac = 1.0 - (update - 1.0) / NUM_UPDATES
        lrnow = frac * optimizer_lr
        optimizer.param_groups[0]["lr"] = lrnow


        print("\n\n\nupdate:{}/{}, collecting data------------------".format(update,NUM_UPDATES))
        for step in tqdm(range(NUM_STEPS)):

            obs[step] = observation
            terminateds[step] = terminated
            
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(observation)

            observation, reward, terminated, _, _ =envs.step(action.cpu().numpy())


            counter += 1
            if terminated == True:
                print("\t\t执行了{}步后结束".format(counter))
                counter = 0

            observation = torch.from_numpy(observation).to(DEVICE)

            reward = torch.tensor(reward,dtype=torch.float32,device=DEVICE)
            terminated = torch.tensor(terminated,dtype=torch.float32,device=DEVICE)

            actions[step] = action
            values[step] = value
            logprobs[step] = logprob
            rewards[step] = reward

        print('done')


        with torch.no_grad():
            next_terminated = terminated
            last_advantage = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    next_no_terminated = 1 - next_terminated
                    next_value = agent.get_value(observation)
                else:
                    next_no_terminated = 1 - terminateds[t+1]
                    next_value = values[t+1]

                current_advantage = rewards[t] + gamma * next_value * next_no_terminated - values[t]
                advantages[t] = last_advantage = current_advantage + gamma * gae_lamda * next_no_terminated * last_advantage

            returns = advantages + values

        print("\n\n\noptimizing---------------------")


        flatten_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        flatten_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        flatten_values = values.reshape(-1)
        flatten_logprobs = logprobs.reshape(-1)
        flatten_advantages = advantages.reshape(-1)
        flatten_returns = returns.reshape(-1)


        indices = np.arange(NUM_STEPS * NUM_ENVS) 
        policy_losses = np.zeros((update_epochs,))
        value_losses = np.zeros((update_epochs,))
        entropy_losses = np.zeros((update_epochs,))
        total_losses = np.zeros((update_epochs,))

        for epoch in range(update_epochs):
            np.random.shuffle(indices)

            for start in range(0, NUM_STEPS, mini_batch_size):
                end = start + mini_batch_size
                choosed_indices = indices[start:end]

                action, new_logprob, entropy, new_value = agent.get_action_and_value(flatten_obs[choosed_indices], flatten_actions[choosed_indices])

                log_ration = new_logprob - flatten_logprobs[choosed_indices]
                ratio = torch.exp(log_ration)

                choosed_advantages = flatten_advantages[choosed_indices]
                choosed_advantages = (choosed_advantages - choosed_advantages.mean()) / (choosed_advantages.std() + 1e-8)


                # Policy loss
                policy_loss_1 = - choosed_advantages * ratio
                policy_loss_2 = - choosed_advantages * torch.clamp(ratio, 1-clip_epsilon , 1+clip_epsilon)
                policy_loss = torch.max(policy_loss_1,policy_loss_2).mean()


                # Value loss
                value_loss = (new_value - flatten_returns[choosed_indices]) ** 2
                value_loss = 0.5 * value_loss.mean()

                #entropy loss
                entropy_loss = entropy.mean()


                loss = policy_loss + 0.5 * value_loss + 0.03 * entropy_loss

                policy_losses[epoch] += policy_loss.item()/mini_batch_num
                value_losses[epoch] += value_loss.item()/mini_batch_num
                entropy_losses[epoch] += entropy_loss.item()/mini_batch_num
                total_losses[epoch] += loss.item()/mini_batch_num
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print("update epoch:{}/{},\ttotal loss:{:.2f}, policy loss:{:.2f}, value loss:{:.2f}, entropy loss:{:.2f}".format(
                epoch, update_epochs, total_losses[epoch], policy_losses[epoch],
                  value_losses[epoch], entropy_losses[epoch]))

        print('\n\n\n\nupdate:{}/{} finished, mean_loss:{:.2f}, mean_policy_loss:{:.2f}, mean_value_loss:{:.2f}, mean_entropy_loss:{:.2f}'.format(
           update, NUM_UPDATES, total_losses.mean(), policy_losses.mean(), value_losses.mean(), entropy_losses.mean()
        ))
        print(agent.actor_logstd)
        print('------------------------------------------------------')


    torch.save(agent.state_dict(), '/home/dukaifeng/my_project/panda_gym_cam-master/my_model/my_model_003.pt')













    # env = gym.make('CarRacing-v2', render_mode='human')
    # observation,_ = env.reset()
    # agent = Agent().to(DEVICE)
    # agent.load_state_dict(torch.load('/home/dukaifeng/my_project/panda_gym_cam-master/my_model/my_model_002.pt'))
    # terminated, truncted = False, False


    # while not terminated and not truncted:
    #     observation = torch.from_numpy(observation).unsqueeze(0).to(DEVICE)
    #     action,_,_,_ = agent.get_action_and_value(observation)
    #     action = np.array(action.squeeze(0).reshape(-1).cpu())

    #     observation, reward, terminated, truncted, info = env.step(action)
    #     time.sleep(1/240)

    # env.close()























