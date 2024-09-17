import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model (download and use the pre-trained YOLOv8 model)
model = YOLO('yolov8n.pt')

# Define vehicle categories
EMERGENCY_VEHICLE_CLASSES = ['ambulance', 'fire_truck', 'police_car']
LIGHT_VEHICLE_CLASSES = ['car', 'motorcycle', 'bicycle']
HEAVY_VEHICLE_CLASSES = ['bus', 'truck']

# Load the traffic camera video feed or dataset
video_path = 'C:\\Users\\korra\\OneDrive\\Desktop\\SIH\\15_minutes_of_heavy_traffic_noise_in_India_｜_14_08_2022_iJZcjZD0fw0.webm'  # Set this to the path of your video file

# Function to classify and count vehicles
def classify_and_count_vehicles(results):
    emergency_vehicle_count = 0
    light_vehicle_count = 0
    heavy_vehicle_count = 0

    for result in results:
        # Check if the result has boxes
        if result.boxes is not None:
            for obj in result.boxes:
                label_index = int(obj.cls[0])  # Class index for the detected object
                label = model.names[label_index]  # Get label using the class index
                
                if label in EMERGENCY_VEHICLE_CLASSES:
                    emergency_vehicle_count += 1
                elif label in LIGHT_VEHICLE_CLASSES:
                    light_vehicle_count += 1
                elif label in HEAVY_VEHICLE_CLASSES:
                    heavy_vehicle_count += 1

    return emergency_vehicle_count, light_vehicle_count, heavy_vehicle_count

# Function to calculate road weight
def calculate_road_weight(emergency_vehicles, light_vehicles, heavy_vehicles):
    # You can assign custom weights to different vehicle types
    return (3 * emergency_vehicles) + (1 * light_vehicles) + (2 * heavy_vehicles)

# Function to optimize the traffic light timer based on road weight
def optimize_traffic_timer(road_weight):
    base_green_light_duration = 30  # in seconds
    
    # Adjust green light duration based on the road weight
    if road_weight > 10:
        green_light_duration = base_green_light_duration + 10
    elif road_weight > 5:
        green_light_duration = base_green_light_duration + 5
    else:
        green_light_duration = base_green_light_duration

    print(f"Optimized green light duration: {green_light_duration} seconds")
    return green_light_duration

# Process video feed and optimize traffic light timer
cap = cv2.VideoCapture(video_path)  # Load the video file

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot read the video file.")
        break
    
    # Perform vehicle detection on the frame using YOLOv8
    results = model(frame)  # This will return a list of Result objects
    
    # Classify and count vehicles
    emergency_vehicles, light_vehicles, heavy_vehicles = classify_and_count_vehicles(results)
    
    # Calculate the road weight
    road_weight = calculate_road_weight(emergency_vehicles, light_vehicles, heavy_vehicles)
    print(f"Detected emergency vehicles: {emergency_vehicles}, light vehicles: {light_vehicles}, heavy vehicles: {heavy_vehicles}")
    print(f"Road weight: {road_weight}")
    
    # Optimize traffic light timer based on road weight
    optimize_traffic_timer(road_weight)
    
    # Display the results (optional)
    annotated_frame = results[0].plot()  # Annotated frame with bounding boxes
    cv2.imshow('YOLOv8 Traffic Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
