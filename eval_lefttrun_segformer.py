import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter   
import torch.nn.functional as F
writer = SummaryWriter('./algo/checkpoints/log')

from env_leftturn_save import LeftTurn
from utils import set_seed, signal_handler
import matplotlib
import csv
matplotlib.use('TkAgg')  # Replace 'Agg' with your desired backend

import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from algo.network_model_dis import Model
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torchvision
# Assuming images are in RGB format and need to be converted to BGR for OpenCV
def save_image(image, path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)

def get_seg(image, model, image_processor, h, w):

    # CITYSCAPES_PALETTE_MAP as provided
    CITYSCAPES_PALETTE_MAP = np.array([
        [0, 0, 0],        # unlabeled
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # pedestrian
        [255, 0, 0],      # rider
        [0, 0, 142],      # Car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
        [110, 190, 160],  # static
        [170, 120, 50],   # dynamic
        [55, 90, 80],     # other
        [45, 60, 150],    # water
        [157, 234, 50],   # road line
        [81, 0, 81],      # ground
        [150, 100, 100],  # bridge
        [230, 150, 140],  # rail track
        [180, 165, 180]   # guard rail
    ], dtype=np.uint8)


    # [1, 3, 512, 512]
    # [batch, channel, h, w]
    print('a')
    print(time.perf_counter())
    inputs = image_processor(images=image, return_tensors="pt")
    print(time.perf_counter())
    model = model.cuda()

    # Move each tensor within 'inputs' to CUDA
    inputs_cuda = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs_cuda)
    print(time.perf_counter())
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    predicted_label = logits.argmax(1)
    rescale = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    print(time.perf_counter())
    predicted_label = rescale(predicted_label)+1
    print(time.perf_counter())
    predicted_label = CITYSCAPES_PALETTE_MAP[predicted_label.cpu()]
    print(time.perf_counter())
    return predicted_label
    
def train_leftturn_task():
    
    set_seed(args.seed)
    
    # construct the DRL agent
    if args.algorithm == 0:
        from algo.TD3PHIL import DRL
        log_dir = 'algo/checkpoints/TD3PHIL.pth'
    elif args.algorithm == 1:
        from algo.TD3IARL import DRL
        log_dir = 'algo/checkpoints/TD3IARL.pth'
    elif args.algorithm == 2:
        from algo.TD3 import DRL
        log_dir = 'algo/checkpoints/TD3HIRL.pth'    
    elif args.algorithm == 3:
        from algo.TD3_dis import DRL as DRL_base    
        #log_dir = 'algo/checkpoints/TD3.pth'
        #log_dir = r'/data/zhuzhuan/VR_driving_data/data/script/leftturn_eye/algo/checkpoints/leftturn/model/noeye/actor1707284410.pkl'
        #log_dir_noeye = r'/data/zhuzhuan/VR_driving_data/data/script/leftturn_eye/algo/checkpoints/leftturn/model/semantic_guide/actor1707910302.pkl'
        #log_dir_noeye = r'/data/zhuzhuan/VR_driving_data/data/script/leftturn_eye/algo/models_leftturn/actor1707552824.pkl'
        
    
    env = LeftTurn(joystick_enabled = args.joystick_enabled, conservative_surrounding = args.simulator_conservative_surrounding, 
               frame=args.simulator_render_frequency, port=args.simulator_port)
               
    #condition = ['noeye','bc','eye']
    condition = ['eye']
    
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    model = model.cuda()
                        
    for cond in condition:
        log_dir = r'/data/zhuzhuan/VR_driving_data/data/script/leftturn_eye/algo/checkpoints/leftturn/model/{}/'.format(cond)
        file_list = os.listdir(log_dir)
        now = datetime.now().strftime("-%d-%m-%Y-%H-%M-%S")
        
        if not os.path.exists(r'/projects/CIBCIGroup/00DataUploading/Zhuoli/eye_tracking_rl/leftturn_town5_segformer/'):
            os.mkdir(r'/projects/CIBCIGroup/00DataUploading/Zhuoli/eye_tracking_rl/leftturn_town5_segformer/')
            
        figure_dir = r'/projects/CIBCIGroup/00DataUploading/Zhuoli/eye_tracking_rl/leftturn_town5_segformer/{}/'.format(cond)
        
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
    
        for ii in range(0,len(file_list)):
            
            
            os.mkdir(os.path.join(figure_dir,file_list[ii])[:-4])
            save_dir = os.path.join(figure_dir,file_list[ii])[:-4]
            
            output_file = "{}_output{}.csv".format(save_dir, now)
            try:
                with open(output_file, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['frame','action','ego_speed','front_speed','ttc','predict_ttc','travel_distance','KL','CC','NSS','SIM','MAE'])                   # Write the variables as a single row in the CSV file
            except IOError as e:
                print("Error:", e)
            
            if not os.path.exists('{}/rgb/'.format(save_dir)):
                os.mkdir('{}/rgb/'.format(save_dir))
            if not os.path.exists('{}/semantic/'.format(save_dir)):
                os.mkdir('{}/semantic/'.format(save_dir))
            if not os.path.exists('{}/machine_att/'.format(save_dir)):
                os.mkdir('{}/machine_att/'.format(save_dir))
            if not os.path.exists('{}/human_att/'.format(save_dir)):
                os.mkdir('{}/human_att/'.format(save_dir))
            
        
            print(file_list)
            log_dir_load = os.path.join(log_dir,file_list[ii])
    
        
            s_dim = [env.observation_size_width, env.observation_size_height]
            a_dim = env.action_size
            
            #DRL_noeye = DRL_base(a_dim, s_dim, device=args.device)
            DRL = DRL_base(a_dim, s_dim, device=args.device)
            
            exploration_rate = args.initial_exploration_rate 
            
            if args.resume and os.path.exists(log_dir_load):
                DRL.load_actor(log_dir_load)
                #DRL_noeye.load_actor(log_dir_noeye)
                start_epoch = 0
            else:
                start_epoch = 0
            
            
            if args.human_model:
                from algo.SL import SL
                SL = SL(a_dim, s_dim)
                SL.load_model('./algo/models_leftturn/SL.pkl')
            
            
            # initialize global variables
            total_step = 0
            a_loss,c_loss, dis_loss, kl_loss = 0,0,0,0
            q_target, cross_correlation, kl_ours, nss,mae, td_errors, sim = 0,0.,0,0,0,0,0
            count_sl = 0
            rgb = None
            
            loss_critic, loss_actor = [], []
            episode_total_reward_list, episode_mean_reward_list = [], []
            global_reward_list, episode_duration_list = [], []
            ttc_list = []
            
            drl_action = [[] for i in range(args.maximum_episode)] # drl_action: output by RL policy
            adopted_action = [[] for i in range(args.maximum_episode)]  # adopted_action: eventual action which may include human guidance action
             
            reward_i_record = [[] for i in range(args.maximum_episode)]  # reward_i: virtual reward (added shaping term);
            reward_e_record = [[] for i in range(args.maximum_episode)]  # reward_e: real reward
            
            
            start_time = time.perf_counter()
            
            for i in range(start_epoch, args.maximum_episode):
                
                # initial episode variables
                ep_reward = 0      
                step = 0
                step_intervened = 0
                done = False
                human_model_activated = False
        
                # initial environment
                observation = env.reset()
                state = np.repeat(np.expand_dims(observation,2), 3, axis=2)
                state_ = state.copy()
            
                while not done:
                    ## Section DRL's actting ##
                    #import pdb; pdb.set_trace()
                    #action, dis = DRL_noeye.choose_action(state)
                    action_pred, risk_pred = DRL.choose_action(state)
                    human_att = DRL.human_att_i.cpu()
                    #action = np.array([0.6])
                    action = action_pred
                    print(action)
                    #action = np.clip( np.random.normal(action,  exploration_rate), -1, 1)
                    
                    #if i < 8:
                    #action = np.array([1.0])
                    #if step > 20:
                    #    action = np.array([-0.5])
                    drl_action[i].append(action)
                    ## End of Section DRL's actting
            
                    
                    ## Section human model's actting ##
                    # use a pre-trained model to output human-like policy which can be taken as human guidance
                    # this model is only activated when the ego agent is approaching risk
                    if args.human_model:
                        if (env.risk is not None ) and (abs(env.risk) <1) and (env.v_low < env.ego_vehicle.get_velocity().y) and (env.index_obs_concerned is not None) and (i % 3 == 1):
                            action = SL.choose_action(observation)
                            env.intervention = True
                            human_model_activated = True
                            env.show_human_model_mode()
                        else:
                            human_model_activated = False
                    ## End of Section human model's actting
                    
                
                    ## Section environment update ##
                    observation_, action_fdbk, reward_e, _, done, scope, risk, travel_dis, rgb,ego_speed,front_speed, semantics  = env.step(action)
                    
                    #state_1 = state_[:,:,2].copy()
                    #state_[:,:,0] = state_[:,:,1].copy()
                    #state_[:,:,1] = state_1
                    #state_[:,:,2] = observation_.copy()
                    
                    
                    #import pdb; pdb.set_trace()
                    #plt.imshow(state_[:,:,0].astype(np.float32))
                    #plt.show()
                    ## End of Section environment update ##
            
            
                    ## Section reward shaping ##
                    # intervention penalty-based shaping
                    if args.reward_shaping == 1:
                        # only the 1st intervened time step is penalized
                        if (action_fdbk is not None) or (human_model_activated is True):
                            if step_intervened == 0:
                                reward_i = -10
                                step_intervened += 1
                            else:
                                reward_i = 0
                        else:
                            reward_i = 0
                            step_intervened = 0
                    # no shaping
                    else:
                        reward_i = 0
                    reward = reward_e + reward_i
                    #reward = reward_e 
                    ## End of Section reward shaping ##
        
        
                    ## Section DRL store ##
                    # real human intervention event occurs
                    if action_fdbk is not None:
                        
                        action = action_fdbk
                        intervention = 1
                        DRL.store_transition(state, action, action_fdbk, intervention, reward, state_)
                        
                        if args.human_model_update:
                            SL.store_transition(observation, action_fdbk)
                            count_sl += 1
        
                    # human model intervention event occurs
                    elif human_model_activated is True:
                        intervention = 1
                        action_fdbk = action
                        DRL.store_transition(state, action, action_fdbk, intervention, reward, state_)
    
                    # no intervention occurs
                    else:
                        intervention = 0
                        DRL.store_transition(state, action, action_fdbk, intervention, reward, state_,risk)
                        #DRL_noeye.store_transition(state, action, action_fdbk, intervention, reward, state_,risk)
                    ## End of DRL store ##
            
            
                    ## Section DRL update ##
                    learn_threshold = args.warmup_threshold if args.warmup else 256
                    #if total_step > learn_threshold:
                    
                    
                    with torch.no_grad():
                        #c_loss, a_loss = DRL.learn(batch_size = 16, epoch=i)
                        #c_loss, a_loss, dis_loss, kl_loss, q_target, cross_correlation, kl_ours, nss, mae, td_errors,sim = DRL_noeye.learn(batch_size = 16, epoch=i, train=False)
                        c_loss, a_loss, dis_loss, kl_loss, q_target, cross_correlation, kl_ours, nss, mae, td_errors, sim = DRL.learn(batch_size = 1, epoch=i,eye=args.eye, train=False)
                        loss_critic.append(np.average(c_loss))
                        loss_actor.append(np.average(a_loss))
                        q_target = q_target.cpu()
                        cross_correlation = cross_correlation.cpu()
                        kl_ours = kl_ours.cpu()
                        #nss = nss.cpu()
                        mae = mae.cpu()
                        
                        # Decrease the exploration rate
                        exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate>args.cutoff_exploration_rate else args.cutoff_exploration_rate
                    ## End of Section DRL update ##
                    
                    try:
                        with open(output_file, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            # Write the variables as a single row in the CSV file
                            #writer.writerow(['frame','action','ego_speed','front_speed','ttc','predict_ttc','KL','CC','NSS','SIM'])  
                            csv_writer.writerow([total_step, action_pred[0],ego_speed,front_speed,risk,risk_pred.cpu().numpy()[0][0]*10,travel_dis,kl_ours.cpu().numpy(),cross_correlation.cpu().numpy()[0],nss,sim,mae.cpu().numpy()])
                    except IOError as e:
                        print("Error:", e)
                    '''
                    model = Model().cuda().eval()
                    checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario2/ckpts/cdnn/model_epoch_10.tar')
                    model.load_state_dict(checkpoint['state_dict'])
                    state_tensor = torch.from_numpy(state_)
                    #import pdb; pdb.set_trace()
                    rgb_image = torch.cat([state_tensor[:,:,0:1],state_tensor[:,:,0:1],state_tensor[:,:,0:1]], axis=2).cuda()
                    
                    rgb_image = rgb_image.transpose(0,2)
                    rgb_image = rgb_image.transpose(1,2)
                    rgb_image = rgb_image.unsqueeze(0).float()
                    new_size = (72,128)
            
                    # Resize/Interpolate the tensor to the new size
                    rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
                    
                    #import pdb; pdb.set_trace()
                    output = model(rgb_image)
                    human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
                    # Calculate the coordinates to place the new image at the center
                    new_height, new_width = 28, 64
                    top = (human_map.size(2) - new_height) // 2
                    left = (human_map.size(3) - new_width) // 2

                    center_cropped_image = human_map[:,:, top:top + new_height, left:left + new_width]
                    # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
                    #import pdb; pdb.set_trace()
                    #center_cropped_image= self.actor.human_map
                    flattened = center_cropped_image.reshape(center_cropped_image.size(0), -1)
   
                    # Apply softmax to get probability distributions
                    prob_distribution = F.softmax(flattened.view(1, -1), dim=1).view_as(flattened)
                    human_att = prob_distribution
                    '''
                    
                    machine_att = DRL.machine_att.cpu()
                    #center_cropped_state = DRL.center_cropped_state.cpu()
                    #human_att = prob_distribution.view(1,1,28, 64)
                    #machine_att_noeye = DRL_noeye.machine_att.cpu()
                    if args.human_model and args.human_model_update and count_sl > 100 and step % 10 == 0:
                        SL.learn(batch_size = 32, epoch = i)
                    new_height, new_width = 28*16, 64*16
                    

                    if rgb is not None:
                        # model

                        #image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
                        #model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
                        # image
                        # (h, w, 3), 0-255
                        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                        #image = Image.open(requests.get(url, stream=True).raw)
                       
                    
                        #path = "/projects/CIBCIGroup/00DataUploading/ChengYou/eccv/leftturn_town5_3car1/with_eye_tracking1/actor1707880391/rgb/0/0011.png"
                        #image = Image.open(path)
                        h, w, _ = np.array(rgb).shape
                    
                        #path_seg = "/projects/CIBCIGroup/00DataUploading/ChengYou/eccv/leftturn_town5_3car1/with_eye_tracking1/actor1707880391/semantic/0/0011.png"
                        #image_seg = Image.open(path_seg)
                        
                        # query
                        # [batch, h, w, 3]
                        #import pdb; pdb.set_trace()
                        #rgb_cuda = torch.from_numpy(rgb.copy()).cuda().float()
                        
                        pred = get_seg(rgb, model, image_processor, h, w)
                        save_segformer = pred[0]
                        pred = pred[0,:,:,0]
                        
                        pred = cv2.resize(pred,(80, 45))
                        pred = np.float16(np.squeeze(pred)/255)
        
                        #pred = torch.from_numpy(pred).cuda()
                        state_1 = state_[:,:,2].copy()
                        state_[:,:,0] = state_[:,:,1].copy()
                        state_[:,:,1] = state_1
                        state_[:,:,2] = pred.copy()
                        #import pdb; pdb.set_trace()
                        #human_att = F.interpolate(human_att, scale_factor=8, mode='bicubic', align_corners=False)
                        #machine_att = F.interpolate(machine_att, scale_factor=8, mode='bicubic', align_corners=False)
                        #machine_att_noeye = F.interpolate(machine_att_noeye, scale_factor=8, mode='bicubic', align_corners=False)
                        # Calculate the coordinates to place the new image at the center
                        top = (rgb.shape[0] - new_height) // 2
                        left = (rgb.shape[1] - new_width) // 2
                        center_cropped_image = rgb[top:top + new_height, left:left + new_width,:]
                        center_cropped_state = semantics[top:top + new_height, left:left + new_width]
                        
                        
                        if not os.path.exists(save_dir + '/rgb/{}'.format(i)):
                            os.mkdir(save_dir + '/rgb/{}'.format(i))
                        if not os.path.exists(save_dir + '/semantic/{}'.format(i)):
                            os.mkdir(save_dir + '/semantic/{}'.format(i))
                        if not os.path.exists(save_dir + '/machine_att/{}'.format(i)):
                            os.mkdir(save_dir + '/machine_att/{}'.format(i))
                        if not os.path.exists(save_dir + '/human_att/{}'.format(i)):
                            os.mkdir(save_dir + '/human_att/{}'.format(i))
                        
                        save_image_path = '{}/rgb/{}/{:04d}.png'.format(save_dir, i, total_step)
                        save_image(center_cropped_image, save_image_path)
                        
                        # For state_[:,:,0], assuming it's grayscale. No need for color conversion.
                        save_image_path = '{}/semantic/{}/{:04d}.png'.format(save_dir, i, total_step)
                        cv2.imwrite(save_image_path, center_cropped_state.astype('uint8'))
                        #import pdb; pdb.set_trace()
                        # Save machine_att[0,0].detach().cpu(), assuming it needs to be converted from tensor to numpy array and possibly normalized
                        machine_att_image = machine_att[0,0].detach().cpu().numpy()
                        machine_att_image = np.clip((machine_att_image-machine_att_image.min())/(machine_att_image.max()-machine_att_image.min()) * 255, 0, 255).astype('uint8')  # Normalize and convert to uint8 if necessary
                        save_image_path = '{}/machine_att/{}/{:04d}.png'.format(save_dir, i,total_step)
                        save_image(machine_att_image, save_image_path)
                        
                        # Save human_att[0,0].detach().cpu(), assuming similar processing as for machine_att
                        #human_att_image = human_att[0,0].detach().cpu().numpy()
                        #human_att_image = np.clip((human_att_image-human_att_image.min())/(human_att_image.max()-human_att_image.min())  * 255, 0, 255).astype('uint8')  # Normalize and convert to uint8 if necessary      
                        human_att_image = save_segformer
                        save_image_path = '{}/human_att/{}/{:04d}.png'.format(save_dir, i, total_step)
                        save_image(human_att_image, save_image_path)
                        
                        '''
                        human_att = F.interpolate(human_att, scale_factor=16, mode='bicubic', align_corners=False)
                        machine_att = F.interpolate(machine_att, scale_factor=16, mode='bicubic', align_corners=False)
                        plt.clf()  # Clear the figure for the next image
                        plt.subplot(2,1,1)
                        plt.imshow(center_cropped_image)
                        #plt.axis('off')  # Hide the axis
                        #plt.savefig('{}/rgb/{:04d}.png.'.format(save_dir, total_step))
                        #plt.clf()  # Clear the figure for the next image
                        
                        #plt.imshow((state_[:,:,0]*255).astype(np.uint8))
                        #plt.axis('off')  # Hide the axis
                        #plt.savefig('{}/semantic/{:04d}.png.'.format(save_dir, total_step))
                        #plt.clf()  # Clear the figure for the next image
    
                        plt.imshow(machine_att[0,0].detach().cpu(),alpha=0.5, cmap='jet')
                        plt.axis('off')  # Hide the axis
                        #plt.savefig('{}/machine_att/{:04d}.png.'.format(save_dir, total_step))
                        #plt.clf()  # Clear the figure for the next image
                        plt.subplot(2,1,2)
                        plt.imshow(center_cropped_image)
                        plt.axis('off')  # Hide the axis
    
                        plt.imshow(human_att[0,0].detach().cpu(),alpha=0.5, cmap='jet')
                        #plt.axis('off')  # Hide the axis
                        plt.show(block=False)
                        plt.pause(0.01)
                        #plt.savefig('{}/human_att/{:04d}.png.'.format(save_dir, total_step))
                        #plt.clf()  # Clear the figure for the next image
                        '''
                    ep_reward += reward
                    global_reward_list.append([reward_e,reward_i])
                    reward_e_record[i].append(reward_e)
                    reward_i_record[i].append(reward_i)
                    
                    adopted_action[i].append(action)
        
                    observation = observation_.copy()
                    state = state_.copy()
                    
                    dura = env.terminate_position
                    total_step += 1
                    step += 1
                    
                    signal.signal(signal.SIGINT, signal_handler)
                
                mean_reward =  ep_reward / step  
                episode_total_reward_list.append(ep_reward)
                episode_mean_reward_list.append(mean_reward)
                episode_duration_list.append(dura)
        
                print('\n episode is:',i)
                print('explore_rate:',round(exploration_rate,4))
                print('c_loss:',round(np.average(c_loss),4))
                print('a_loss',round(np.average(a_loss),4))
                print('total_step:',total_step)
                print('episode_step:',step)
                print('episode_cumu_reward',round(ep_reward,4))
                print('travel distance',travel_dis)
                ttc_list = np.array(ttc_list)
                print('ttc', np.average(ttc_list[ttc_list>0]))
                print('nss', np.average(nss))
                print('kl_ours', np.average(kl_ours))
                print('sim', np.average(sim))
                print('cc', np.average(cross_correlation))
        
                writer.add_scalar('reward/reward_episode', ep_reward, ii*args.maximum_episode+i)
                writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), ii*args.maximum_episode+i)
                writer.add_scalar('reward/risk', risk, ii*100+i)
                writer.add_scalar('reward/duration_episode', step, ii*args.maximum_episode+i)
                writer.add_scalar('reward/survival_distance', travel_dis, ii*args.maximum_episode+i)
                writer.add_scalar('rate_exploration', round(exploration_rate,4), ii*args.maximum_episode+i)
                writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), ii*args.maximum_episode+i)
                writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), ii*args.maximum_episode+i)
                writer.add_scalar('loss/loss_distance', dis_loss, ii*args.maximum_episode+i)
                writer.add_scalar('loss/loss_kl', kl_loss, ii*args.maximum_episode+i)
                writer.add_scalar('loss/loss_sim', np.average(sim), ii*args.maximum_episode+i)
                writer.add_scalar('loss/q_target', np.average(q_target), ii*args.maximum_episode+i)
                writer.add_scalar('loss/td_errors', np.average(td_errors), ii*args.maximum_episode+i)
                writer.add_scalar('loss/cross_correlation', np.average(cross_correlation), ii*args.maximum_episode+i)
                writer.add_scalar('loss/kl_ours', np.average(kl_ours), ii*args.maximum_episode+i)
                writer.add_scalar('loss/nss', np.average(nss), ii*args.maximum_episode+i)
                writer.add_scalar('loss/mae', np.average(mae), ii*args.maximum_episode+i)
    
        print('total time:',time.perf_counter()-start_time)        
        
        timeend = round(time.time())
        #DRL.save_actor('./algo/models_leftturn', timeend)
        
    pygame.display.quit()
    pygame.quit()
    
    action_drl = drl_action[0:i]
    action_final = adopted_action[0:i]
        #scio.savemat('dataleftturn_{}-{}-{}.mat'.format(args.seed,args.algorithm,timeend), mdict={'action_drl': action_drl,'action_final': action_final,
        #                                                'stepreward':global_reward_list,'mreward':episode_mean_reward_list,
        #                                                'step':episode_duration_list,'reward':episode_total_reward_list,
        #                                                'r_i':reward_i_record,'r_e':reward_e_record})
        
    


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
    parser.add_argument('--human_model', action="store_true", help='whehther to use human behavior model (default: False)', default=False)
    parser.add_argument('--human_model_update', action="store_true", help='whehther to update human behavior model (default: False)', default=False)
    parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:400)', default=20)
    parser.add_argument('--seed', type=int, help='fix random seed', default=2)
    parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
    parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
    parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
    parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
    parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
    parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 1e4)', default=1e4)
    parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:proposed) (default: 1)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
    parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 12)', default=12)
    parser.add_argument('--simulator_conservative_surrounding', action="store_true", help='surrounding vehicles are conservative or not (default: False)', default=False)
    parser.add_argument('--joystick_enabled', action="store_true", help='whether use Logitech G29 joystick for human guidance (default: False)', default=False)
    parser.add_argument('--eye', action="store_true", default=False)
    args = parser.parse_args()

    # Run
    train_leftturn_task()
