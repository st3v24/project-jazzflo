import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt 

pipe= rs.pipeline()
cfg= rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)

p=pipe.start(cfg)

while True:
    frame=pipe.wait_for_frames()
    depth_frame=frame.get_depth_frame()
    color_frame=frame.get_color_frame()
    
    depth_image= np.asanyarray(depth_frame.get_data())
    color_image= np.asanyarray(color_frame.get_data())
    
    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_image)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.imshow(color)
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    plt.imshow(colorized_depth)
    align = rs.align(rs.stream.color)
    frameset = align.process(frame)
    #plt.show()

# Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

# Show the two frames together:
    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    depth = np.asanyarray(aligned_depth_frame.get_data())
    #enter location of x and y points as the  300,300 values
    depth = depth[300,300].astype(float)

# Get data scale from the device and convert to meters
    depth_scale = p.get_device().first_depth_sensor().get_depth_scale()
    depth = depth * depth_scale
    dist,_,_,_ = cv2.mean(depth)
    print("Detected a {0} {1:.3} m away.".format("--", depth))
pipe.stop() 