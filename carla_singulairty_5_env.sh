#!/bin/bash

# Open a new gnome-terminal and run the first command
gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.14.sif /home/carla/CarlaUE4.sh -carla-port=2000; exec bash" 

gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.14.sif /home/carla/CarlaUE4.sh -carla-port=3000; exec bash" 

gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.14.sif /home/carla/CarlaUE4.sh -carla-port=3500; exec bash" 

gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.14.sif /home/carla/CarlaUE4.sh -carla-port=4500; exec bash" 

gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.14.sif /home/carla/CarlaUE4.sh -carla-port=5000; exec bash" 