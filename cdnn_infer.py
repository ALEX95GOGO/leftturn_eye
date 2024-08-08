import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import cv2
from algo.network_model_dis import Model
import torch.nn.functional as F
import imageio as io
import time

# Assuming images are in RGB format and need to be converted to BGR for OpenCV
def save_image(image, path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)
    
    
if __name__ == '__main__':
    model = Model().cuda().eval()
    checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario2/ckpts/cdnn/model_epoch_2.tar')
    save_dir = r'/projects/CIBCIGroup/00DataUploading/Zhuoli/eye_tracking_rl'
    model.load_state_dict(checkpoint['state_dict'])
    #state_tensor = torch.from_numpy(state_)
    #import pdb; pdb.set_trace()
    #rgb_image = torch.cat([state_tensor[:,:,0:1],state_tensor[:,:,0:1],state_tensor[:,:,0:1]], axis=2).cuda()
    
    condition = ['bc', 'eye', 'noeye']
    seg_dir = r'/projects/CIBCIGroup/00DataUploading/Zhuoli/eye_tracking_rl/leftturn_fixed_action/'
    
    for cond in condition:
        cond_path = os.path.join(seg_dir, cond)
        
        actor_list = os.listdir(cond_path)
        
        for k in range(5,10):
            for kk in range(10):
                seg_fig_dir = os.path.join(cond_path, actor_list[k], 'semantic',str(kk))
                
                print(seg_fig_dir)
                figures = os.listdir(seg_fig_dir)
                
                for ii in range(len(figures)):
                    rgb_image = io.imread(os.path.join(seg_fig_dir,figures[ii]), mode='L')
                    #rgb_image = cv2.resize(rgb_image, (384, 216), interpolation=cv2.INTER_CUBIC)
                    #import pdb; pdb.set_trace()
                    rgb_image = rgb_image.astype('float32')/255.0
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
                    #rgb_image = rgb_image[:,::-1]
                    
                    rgb_image = rgb_image.transpose(2, 0, 1)
                    rgb_image = np.ascontiguousarray(rgb_image)
                    rgb_image = torch.from_numpy(rgb_image)
                    
                    rgb_image = rgb_image.unsqueeze(0).float().cuda()
                    #new_size = (72,128)
                    new_size = (56*2,128*2)
                    # Resize/Interpolate the tensor to the new size
                    rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
                    
                    with torch.no_grad():
                        #print(time.perf_counter())
                        input_time = time.perf_counter()
                        #for i in range(10000):
                        output = model(rgb_image)
                        print(time.perf_counter() - input_time)
                    #import pdb; pdb.set_trace()
                    '''
                    output_image = output[0, 0].cpu().detach().numpy()
                    input_image = rgb_image[0].cpu().detach().numpy()
                    #import pdb; pdb.set_trace()
                    # Apply a colormap to the output image
                    colormap = plt.get_cmap('viridis')  # Choose your desired colormap
                    output_image = (output_image-output_image.min())/(output_image.max()-output_image.min())
                    mapped_output = colormap(output_image)[:,:,:-1]
                    
                    # Normalize input image to the range [0, 1]
                    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
                    input_image = input_image.transpose(1, 2, 0)
                  
                    # Blend the two images (adjust the alpha value as needed for transparency)
                    alpha = 0.5  # Adjust this value for the desired level of transparency
                    overlay_image = alpha * mapped_output + (1 - alpha) * input_image
                    
                    #import pdb; pdb.set_trace()
                    # Display the overlay
                    plt.imshow(overlay_image, cmap=colormap)
                    plt.show()
                    '''
                    human_map = F.interpolate(output, size=(28, 64), mode='bilinear',align_corners=False).clone()
                    # Calculate the coordinates to place the new image at the center
                    #new_height, new_width = 28, 64
                    #top = (human_map.size(2) - new_height) // 2
                    #left = (human_map.size(3) - new_width) // 2
                    #center_cropped_image = human_map[:,:, top:top + new_height, left:left + new_width]
                    center_cropped_image = human_map
                    # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
                    #import pdb; pdb.set_trace()
                    #center_cropped_image= self.actor.human_map
                    flattened = center_cropped_image.reshape(center_cropped_image.size(0), -1)
                    
                    # Apply softmax to get probability distributions
                    prob_distribution = F.softmax(flattened.view(1, -1), dim=1).view_as(flattened)
                    human_att = prob_distribution.view(28,64)
                    
                    human_att_image = human_att.cpu().numpy()
                    human_att_image = np.clip((human_att_image-human_att_image.min())/(human_att_image.max()-human_att_image.min())  * 255, 0, 255).astype('uint8')  # Normalize and convert to uint8 if necessary
                    #import pdb; pdb.set_trace()
                    if not os.path.exists(os.path.join(cond_path, actor_list[k],r'{}/human_att_fixed'.format(save_dir))):
                        os.mkdir(os.path.join(cond_path, actor_list[k],r'{}/human_att_fixed'.format(save_dir)))
                    if not os.path.exists(os.path.join(cond_path, actor_list[k],r'{}/human_att_fixed/{}'.format(save_dir,kk))):
                        os.mkdir(os.path.join(cond_path, actor_list[k],r'{}/human_att_fixed/{}'.format(save_dir,kk)))
                    save_image_path = os.path.join(cond_path, actor_list[k],'human_att_fixed/{}/{}'.format(kk,figures[ii]))
                    save_image(human_att_image, save_image_path)
