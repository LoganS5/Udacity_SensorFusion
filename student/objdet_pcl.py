# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # step 2 : create instance of open3d point-cloud class
    pc3d = o3d.geometry.PointCloud()
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pc3d.points = o3d.utility.Vector3dVector(pcl[:,0:3])
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    # since the other instructions seem to indicate that the visualizer should be recreated each call, I've done that and added geometry each call
    vis.add_geometry(pc3d)
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    def close_vis(visualizer):
        visualizer.close()
    vis.register_key_callback(262, close_vis)
    vis.run()
    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar_data = [laser for laser in frame.lasers if laser.name == lidar_name][0]
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar_data.ri_return1.range_image_compressed)) #ri cell format = [range, intensity, elongation, is in any no label zone]
    ri = np.array(ri.data).reshape(ri.shape.dims)
    # step 2 : extract the range and the intensity channel from the range image
    ri_range = np.array(ri[:,:,0])
    ri_intensity = np.array(ri[:,:,1])
    # step 3 : set values <0 to zero
    ri_range[ri_range<0] = 0.0
    ri_intensity[ri_intensity<0] = 0.0
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range_max, range_min = np.amax(ri_range), np.amin(ri_range)
    ri_range = (ri_range - range_min)/(range_max - range_min) * 255
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    intensity_cap_max, intensity_cap_min = np.percentile(ri_intensity,99), np.percentile(ri_intensity,1)
    ri_intensity[ri_intensity>intensity_cap_max] = intensity_cap_max
    ri_intensity[ri_intensity<intensity_cap_min] = intensity_cap_min
    ri_intensity = (ri_intensity - intensity_cap_min)/(intensity_cap_max - intensity_cap_min) * 255
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((ri_range, ri_intensity)).astype(np.uint8)
    # extra step from rubric: Crop range image to +/- 90 deg. left and right of the forward-facing x-axis
    width = img_range_intensity.shape[1]
    img_range_intensity = img_range_intensity[:,int(width*1/4):int(width*3/4)]
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    cell_height = (configs.lim_x[1]-configs.lim_x[0])/configs.bev_height
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:,0] = np.floor((lidar_pcl[:,0]-configs.lim_x[0])/(cell_height+0.0000001))
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    cell_width = (configs.lim_y[1]-configs.lim_y[0])/configs.bev_width
    lidar_pcl_cpy[:,1] = np.floor((lidar_pcl[:,1]-configs.lim_y[0])/(cell_width+0.0000001))
    
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy)

    # create another copy for sorting height
    lidar_pcl_cpy2 = np.copy(lidar_pcl_cpy)
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    # I'm assuming -z is supposed to be -intensity (see https://knowledge.udacity.com/questions/870593 https://knowledge.udacity.com/questions/849169)
    order = np.lexsort((-lidar_pcl_cpy[:,3], lidar_pcl_cpy[:,1], lidar_pcl_cpy[:,0]))
    lidar_pcl_cpy = lidar_pcl_cpy[order]
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, idx, counts = np.unique(lidar_pcl_cpy[:,0:2], return_index=True, return_counts=True, axis=0)
    lidar_bright_pcl = lidar_pcl_cpy[idx]
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud

    # The method of normalising asked for, which I'm assuming is based on the method from the course:
    # ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    # doesn't make sense to me as you end up with values > 255, which turn into essentially random values between 0 and 255 during uint8 conversion
    # (you can see this happening in the solution file where it is originally implemented)
    # I've assumed there's an error in the content with that, and the percentile approach works, so I have done that below.
    # (I've also asked this here, but haven't yet recieved an answer https://knowledge.udacity.com/questions/870589)

    # lidar_bright_pcl[:,3] = (np.amax(lidar_bright_pcl[:,3])-np.amin(lidar_bright_pcl[:,3])) * lidar_bright_pcl[:,3] * 255 / (np.amax(lidar_bright_pcl[:,3]) - np.amin(lidar_bright_pcl[:,3]))
    intensity_cap_max, intensity_cap_min = np.percentile(lidar_bright_pcl[:,3],98), np.percentile(lidar_bright_pcl[:,3],2)
    lidar_bright_pcl[:,3][lidar_bright_pcl[:,3]>intensity_cap_max] = intensity_cap_max
    lidar_bright_pcl[:,3][lidar_bright_pcl[:,3]<intensity_cap_min] = intensity_cap_min
    lidar_bright_pcl[:,3] = (lidar_bright_pcl[:,3] - intensity_cap_min)/(intensity_cap_max - intensity_cap_min)
    
    for el in lidar_bright_pcl:
        intensity_map[int(el[0]), int(el[1])] = el[3]


    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imshow('intensity bev',intensity_map)
    
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    # I'm computing lidar_pcl_top here, as above I used lidar_pcl_bright
    order = np.lexsort((-lidar_pcl_cpy2[:,2], lidar_pcl_cpy2[:,1], lidar_pcl_cpy2[:,0]))
    lidar_pcl_cpy2 = lidar_pcl_cpy2[order]
    _, idx = np.unique(lidar_pcl_cpy2[:,0:2], return_index=True, axis=0)
    lidar_pcl_top = lidar_pcl_cpy2[idx]
    lidar_pcl_top[:,2] = lidar_pcl_top[:,2]/(configs.lim_z[1]-configs.lim_z[0])
    for el in lidar_pcl_top:
        height_map[int(el[0]), int(el[1])] = el[2]
    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imshow('height_map', height_map)

    #######
    ####### ID_S2_EX3 END #######       

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


