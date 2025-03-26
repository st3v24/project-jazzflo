import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Load YOLOv7 model
def load_model():
    weights = "trained_weights/yolov7_best_v3.pt"
    model = torch.hub.load("yolov7", "custom", weights, source="local", trust_repo=True)
    return model

def class_map(class_id):
    class_names = {0: "Bud", 1: "Flower"}
    return class_names.get(class_id, "Unknown")

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
p = pipe.start(cfg)

# Initialize YOLOv7 model
model = load_model()

# Create alignment object
align = rs.align(rs.stream.color)

# Create colorizer for depth visualization
colorizer = rs.colorizer()

try:
    while True:
        # Wait for frames
        frame = pipe.wait_for_frames()
        
        # Align frames
        aligned_frames = align.process(frame)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Get depth scale for distance calculation
        depth_scale = p.get_device().first_depth_sensor().get_depth_scale()
        
        # Run YOLOv7 inference on color image
        results = model(color_image)
        detections = results.pandas().xyxy[0]
        
        # Create visual output image (copy of color image)
        display_image = color_image.copy()
        
        # Process each detection
        for idx, detection in detections.iterrows():
            # Extract detection information
            x_min, y_min, x_max, y_max = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            class_id = int(detection['class'])
            confidence = detection['confidence']
            class_name = class_map(class_id)
            
            # Calculate center point of detection
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            
            # Get depth at center point (in meters)
            depth_value = depth_image[center_y, center_x] * depth_scale
            
            # Skip if depth is invalid (zero)
            if depth_value <= 0:
                continue
                
            # Draw bounding box and information
            color = (0, 255, 0) if class_id == 1 else (0, 0, 255)  # Green for flowers, Red for buds
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw center point
            cv2.circle(display_image, (center_x, center_y), 4, (255, 0, 255), -1)
            
            # Add text with class, confidence and depth
            label = f"{class_name}: {confidence:.2f}, {depth_value:.3f}m"
            cv2.putText(display_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"Detected {class_name} {depth_value:.3f}m away at position ({center_x}, {center_y})")
        
        # Create colorized depth image for display
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        # Stack images side by side
        images_horizontal = np.hstack((display_image, colorized_depth))
        
        # Display the images
        cv2.imshow('YOLOv7 Detections with Depth', images_horizontal)
        
        # Break on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
            
except Exception as e:
    print(f"Error: {e}")
    
finally:
    # Stop pipeline
    pipe.stop()
    cv2.destroyAllWindows()