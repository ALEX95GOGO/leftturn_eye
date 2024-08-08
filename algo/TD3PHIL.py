
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cpprb import PrioritizedReplayBuffer

from algo.network_model import Actor,Critic
from algo.util import hard_update, soft_update


MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
GAMMA = 0.95
LR_C = 0.0005
LR_A = 0.0002
LR_I = 0.01
TAU = 0.001
POLICY_NOSIE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5

'''
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.ch_in = 2
        self.conv1_ch = 16                                                 
        self.conv2_ch = 20
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = 80                                                        
        self.W = 80
        self.node_size = 36                                                
        self.lin_hid = 100
        self.out_dim = 1
        self.sp_coord_dim = 2
        #self.N = int(16**2)      
        #self.N = int(44*44) 
        #self.N = int(30*30)  
        self.N = 390
        #self.N = int(4096) 
        #self.N = int(44*86)    
        #self.N = int(17836)                                
 
        self.conv1 = nn.Conv2d(self.ch_in,self.conv1_ch,kernel_size=(5,5))
        self.conv2 = nn.Conv2d(self.conv1_ch,self.conv2_ch,kernel_size=(5,5))
        self.conv3 = nn.Conv2d(self.conv2_ch,self.conv3_ch,kernel_size=(5,5),stride=2)
        self.conv4 = nn.Conv2d(self.conv3_ch,self.conv4_ch,kernel_size=(5,5))
        
        self.proj_shape = (self.conv4_ch+self.sp_coord_dim,self.node_size)  
        #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        
        self.linear1 = nn.Linear(self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size],
     elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)
    def forward(self,x):
            N, Cin, H, W = x.shape
            #import pdb; pdb.set_trace()
            x = self.conv1(x) 
            x = torch.relu(x)
            x = self.conv2(x) 
            #x = x.squeeze() 
            x = torch.relu(x) 
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.conv4(x)
            x = torch.relu(x)
            #import pdb; pdb.set_trace()
            if x.dim() == 3:
                # If x has 3 dimensions, add a batch dimension
                x = x.unsqueeze(0)
            _,_,cH,cW = x.shape
            xcoords = torch.arange(cW).repeat(cH,1).float() / cW               
            ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
            spatial_coords = torch.stack([xcoords,ycoords],dim=0).cuda()
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N,1,1,1) 
            x = torch.cat([x,spatial_coords],dim=1)
            x = x.permute(0,2,3,1)
            x = x.flatten(1,2)
            
            K = self.k_proj(x)                                                 
            K = self.k_norm(K) 
            
            Q = self.q_proj(x)
            Q = self.q_norm(Q) 
            
            V = self.v_proj(x)
            V = self.v_norm(V) 
            A = torch.einsum('bfe,bge->bfg',Q,K)                               
            A = A / np.sqrt(self.node_size)
            A = torch.nn.functional.softmax(A,dim=2) 
            #import pdb; pdb.set_trace()
            with torch.no_grad():
                self.att_map = A.clone()
            E = torch.einsum('bfc,bcd->bfd',A,V)                               
            E = self.linear1(E)
            E = torch.relu(E)
            E = self.norm1(E)  
            E = E.max(dim=1)[0]
            y = self.linear2(E)  
            y = torch.nn.functional.log_softmax(y,dim=1)
            return y

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.ch_in = 2
        self.conv1_ch = 16                                                 
        self.conv2_ch = 20
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = 80                                                        
        self.W = 80
        self.node_size = 36                                                
        self.lin_hid = 100
        self.out_dim = 1
        self.sp_coord_dim = 2
        #self.N = int(16**2)      
        #self.N = int(44*44) 
        #self.N = int(30*30)  
        self.N = 390
        #self.N = int(44*86)    
        #self.N = int(17836)                                
 
        self.conv1 = nn.Conv2d(self.ch_in,self.conv1_ch,kernel_size=(5,5))
        self.conv2 = nn.Conv2d(self.conv1_ch,self.conv2_ch,kernel_size=(5,5))
        self.conv3 = nn.Conv2d(self.conv2_ch,self.conv3_ch,kernel_size=(5,5) ,stride=2)
        self.conv4 = nn.Conv2d(self.conv3_ch,self.conv4_ch,kernel_size=(5,5))
        
        self.proj_shape = (self.conv4_ch+self.sp_coord_dim,self.node_size)  
        #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        
        self.linear1 = nn.Linear(self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size],
     elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size+1, self.out_dim)
    def forward(self,x, action):
            N, Cin, H, W = x.shape
            #import pdb; pdb.set_trace()
            x = self.conv1(x) 
            x = torch.relu(x)
            x = self.conv2(x) 
            #x = x.squeeze() 
            x = torch.relu(x) 
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.conv4(x)
            x = torch.relu(x)
            #import pdb; pdb.set_trace()
            if x.dim() == 3:
                # If x has 3 dimensions, add a batch dimension
                x = x.unsqueeze(0)
            _,_,cH,cW = x.shape
            xcoords = torch.arange(cW).repeat(cH,1).float() / cW               
            ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
            spatial_coords = torch.stack([xcoords,ycoords],dim=0).cuda()
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N,1,1,1) 
            x = torch.cat([x,spatial_coords],dim=1)
            x = x.permute(0,2,3,1)
            x = x.flatten(1,2)
            
            K = self.k_proj(x)                                                 
            K = self.k_norm(K) 
            
            Q = self.q_proj(x)
            Q = self.q_norm(Q) 
            
            V = self.v_proj(x)
            V = self.v_norm(V) 
            A = torch.einsum('bfe,bge->bfg',Q,K)                               
            A = A / np.sqrt(self.node_size)
            A = torch.nn.functional.softmax(A,dim=2) 
            #import pdb; pdb.set_trace()
            with torch.no_grad():
                self.att_map = A.clone()
            E = torch.einsum('bfc,bcd->bfd',A,V)                               
            E = self.linear1(E)
            E = torch.relu(E)
            E = self.norm1(E)  
            E = E.max(dim=1)[0]
            E = torch.cat([E, action], 1)
            y = self.linear2(E)  
            y = torch.nn.functional.log_softmax(y,dim=1)
            return y
'''
            
class DRL:
        
    def __init__(self, action_dim, state_dim, device='cuda', LR_C = LR_C, LR_A = LR_A):

        self.device = device
        
        self.state_dim = state_dim[0] * state_dim[1] * 2
        self.state_dim_width = state_dim[0]
        self.state_dim_height = state_dim[1]
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOSIE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0

        self.pointer = 0
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY,
                                                  {"obs": {"shape": (45,80,3)},
                                                   "act": {"shape":action_dim},
                                                   "acte": {"shape":action_dim},
                                                   "intervene": {},
                                                   "rew": {},
                                                   "next_obs": {"shape": (45,80,3)},
                                                   "done": {}},
                                                  next_of=("obs"))
        
        self.actor = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR_A)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, 0.996)
        self.previous_epoch = 0
        
        self.critic = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(),LR_C)
        
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)
        
            
    def learn(self, batch_size = BATCH_SIZE, epoch=0):

        ## batched state, batched action, batched action from human, batched human intervention signal, batched reward, batched next state
        data = self.replay_buffer.sample(batch_size)
        idxs = data['indexes']
        states, actions, actions_h = data['obs'], data['act'], data['acte']
        interv, rewards = data['intervene'], data['rew']
        next_states, dones = data['next_obs'], data['done']
        
        states = torch.FloatTensor(states).permute(0,3,1,2).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        actions_h = torch.FloatTensor(actions_h).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0,3,1,2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # initialize the loss variables
        loss_c, loss_a = 0, 0

        ## calculate the predicted values of the critic
        with torch.no_grad():
            noise1 = (torch.randn_like(actions) * self.policy_noise).clamp(0, 1)
            next_actions = (self.actor_target(next_states).detach() + noise1).clamp(0, 1)
            target_q1, target_q2 = self.critic_target([next_states, next_actions])
            target_q = torch.min(target_q1,target_q2)
            y_expected = rewards + (1-dones)*self.gamma * target_q    
        y_predicted1, y_predicted2 = self.critic.forward([states, actions]) 
        
        ## calculate td error
        td_errors = abs(y_expected - y_predicted1.detach())
        
        ## update the critic
        loss_critic = F.mse_loss(y_predicted1,y_expected) + F.mse_loss(y_predicted2,y_expected)
        criterion = nn.BCELoss().cuda()
        #import pdb; pdb.set_trace()
        # Define the new size you want to crop to
        new_height, new_width = 28, 64
        
        # Calculate the coordinates to place the new image at the center
        top = (self.actor.human_map.size(2) - new_height) // 2
        left = (self.actor.human_map.size(3) - new_width) // 2
        
        # Perform the crop
        center_cropped_image = self.actor.human_map[:,:, top:top + new_height, left:left + new_width]
        # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
        #import pdb; pdb.set_trace()
        flattened = center_cropped_image.reshape(center_cropped_image.size(0), -1)
        
        # Apply softmax across the spatial dimension (which is now flattened)
        prob_distribution = F.softmax(flattened, dim=1)
        
        # Reshape back to the original spatial dimensions [C, H, W]
        prob_distribution = prob_distribution.view(center_cropped_image.size())

        # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
        flattened_actor = self.actor.padded_saliency.cuda().unsqueeze(0).unsqueeze(0).reshape(center_cropped_image.size(0), -1)
        
        # Apply softmax across the spatial dimension (which is now flattened)
        prob_distribution_actor = F.softmax(flattened_actor, dim=1)
        
        # Reshape back to the original spatial dimensions [C, H, W]
        prob_distribution_actor = prob_distribution_actor.view(center_cropped_image.size())


        bce_loss = criterion(prob_distribution_actor, prob_distribution)
        #import pdb; pdb.set_trace()
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()

        ## update the actor
        if self.itera % self.policy_freq == 0:
            
            ## select those human guided data index
            index_imi, _ = np.where(interv==1)
            states_imi = states[index_imi]
            actions_imi = actions[index_imi]
            pred_actions = self.actor.forward(states)
            
            if len(index_imi) > 0:
                ## calculate the behavior-cloning (BC) objective for imitating human actions when aviliable.
                imitation_loss = 3 * ((self.actor.forward(states_imi) - actions_imi)**2).sum()
                ## calculate q-advantage metric
                with torch.no_grad():
                    q_adv = torch.exp( self.critic_target([states, actions])[0] - self.critic_target([states, pred_actions])[0])
                    q_weight = torch.zeros_like(q_adv)
                    q_weight[index_imi] = 1
                    qa_errors = q_adv*q_weight

            else:
                imitation_loss = 0.
                qa_errors = 0.
            
            loss_actor = -self.critic([states, pred_actions])[0] + imitation_loss + bce_loss*10
            loss_actor = loss_actor.mean()
            #print(-self.critic([states, pred_actions])[0])
            #print(imitation_loss)
            print(bce_loss)
            
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            if epoch != self.previous_epoch:
                self.actor_scheduler.step() 
            self.previous_epoch = epoch
            
            soft_update(self.actor_target,self.actor,self.tau)
            soft_update(self.critic_target,self.critic,self.tau)

            loss_a = loss_actor.item()

        loss_c = loss_critic.item()
        
        self.itera += 1
        
        # TDQA priority calculation
        priorities = td_errors + qa_errors
        priorities = priorities.cpu().numpy()

        self.replay_buffer.update_priorities(idxs, priorities)

        return loss_c, loss_a
    
                
    def choose_action(self,state):

        state = torch.FloatTensor(state).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
        
        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action,-1, 1)

        return action
    

    def store_transition(self,  s, a, ae, i, r, s_, d=0):
        self.replay_buffer.add(obs=s,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               done=d)
    

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
        
    
    def load_model(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
    
    def load_actor(self, output):
        self.actor.load_state_dict(torch.load(output))
        
    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
    
    def save_actor(self, output, no):
        torch.save(self.actor.state_dict(), '{}/actor{}.pkl'.format(output, no))
    
    def save(self, log_dir, epoch):
        state = {'actor':self.actor.state_dict(), 'actor_target':self.actor_target.state_dict(),
                 'actor_optimizer':self.actor_optimizer.state_dict(), 
                 'critic':self.critic.state_dict(), 'critic_target':self.critic_target.state_dict(),
                 'critic_optimizers':self.critic_optimizers.state_dict(),
                 'epoch':epoch}
        torch.save(state, log_dir)
        

    def load(self, log_dir):
        checkpoint = torch.load(log_dir)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizers.load_state_dict(checkpoint['critic_optimizers'])
        
        
        
        
        
        
        
