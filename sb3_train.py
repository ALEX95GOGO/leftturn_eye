from stable_baselines3 import PPO, SAC
from env_leftturn_gym import LeftTurn
import torch as torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#from stable_baselines3.common.callbacks import TensorBoardCallback
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
            
class Model(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)


        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 4*n)

        self.convu3 = conv3x3(8*n, 4*n)
        self.convu2 = conv3x3(6*n, 2*n)
        self.convu1 = conv3x3(3*n, 1*n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        #import pdb; pdb.set_trace()
       
        x1 = x
        x1 = self.convd1(x1)
        # print(x1.size())

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)
        # print(x2.size())

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)
        # print(x3.size())

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        # print(x4.size())

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)
        # print(y3.size())

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)
        # print(y2.size())

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)
        # print(y1.size())

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
        #y1 = F.softmax(y1, dim=1)
        return y1

        
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        nb_actions = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, 6),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 6),
            nn.MaxPool2d(2),
            #nn.Flatten(),
        )
        
        #256x19x8
        #self.conv1 = nn.Conv2d(n_input_channels, 6, 6)
        #self.conv2 = nn.Conv2d(6, 16, 6)
        
        # Compute shape by doing one forward pass
        #with torch.no_grad():
        #    n_flatten = self.cnn(
        #        torch.as_tensor(observation_space.sample()[None]).float()
        #    ).flatten().shape[1]


        #self.fc1 = nn.Linear(16*16*7, 256)
        #self.fc1 = nn.Linear(39216, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, nb_actions)
        #self.sig = nn.Tanh()
        features_dim = 128
        self.linear = nn.Sequential(nn.Linear(112*18, features_dim), nn.ReLU())
        #self.linear = nn.Sequential(nn.Linear(306*34, features_dim), nn.ReLU())
        
        self.node_size = 12
        self.proj_shape = (18,self.node_size)  
        #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        self.N = 16*7
        #self.N = 34*9
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            self.model = Model().cuda().eval()
            checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario1/ckpts/cdnn/model_epoch_2.tar')
            self.model.load_state_dict(checkpoint['state_dict'])
    
            #self.model = Model().cuda().eval()
            #checkpoint = torch.load('/data/zhuzhuan/CDNN-traffic-saliency/ckpts/cdnn/model_epoch_26.tar')
            #checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/eye_TD3/algo/checkpoints/car_following/model/eye/actor1707651533.pkl')
            #import pdb; pdb.set_trace()
            #self.model.load_state_dict(checkpoint['model'])
            rgb_image = torch.cat([observations[:,1:2],observations[:,1:2],observations[:,1:2]], axis=1)
            # Desired new size
            #new_size = (216//2, 384//2)
            new_size = (72,128)
            #import pdb; pdb.set_trace()
            # Resize/Interpolate the tensor to the new size
            rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
            self.observation = observations[0,0]
            
            output = self.model(rgb_image)
            self.human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
        
        #x = inp.unsqueeze(0) if len(inp.shape)==3 else inp
        N, Cin, H, W = observations.shape
        #import pdb; pdb.set_trace()
        #x = F.max_pool2d( self.conv1(x), 2)
        #x = F.max_pool2d( self.conv2(x), 2)
        x = self.cnn(observations)
        
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
        #print(A.shape)
        self.att = A
        E = torch.einsum('bfc,bcd->bfd',A,V)                               
                	
        x = x.reshape(E.size(0),-1)
        
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.sig(self.fc4(x))
        return self.linear(x)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


tensorboard_log_dir = "./tensorboard_logs_ppo/"
#tensorboard_callback = TensorBoardCallback(verbose=1)

env = LeftTurn(port=9002)  # Replace with your environment's class name
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir)
model.learn(total_timesteps=500000)

'''
env = LeftTurn()  # Replace with your environment's class name
model = SAC("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
'''