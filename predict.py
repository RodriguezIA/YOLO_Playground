import os
from ultralytics import YOLO

# model_path = os.path.join(os.getcwd(), 'runs', 'train27', 'weights', 'best.pt')
# images_path = os.path.join(os.getcwd(), 'pcb', 'train', 'images')
# images_path_lego = os.path.join(os.getcwd(), 'images_inference')
# lego_model_path = os.path.join('runs', 'detect', 'train33', 'weights', 'best.pt')

images_inference_path = os.path.join(os.getcwd(), 'images_inference')


model = YOLO('pcb.pt')
# model = YOLO(lego_model_path)

image_files = [
    os.path.join(images_inference_path, 'pcb1.jpg'),
    os.path.join(images_inference_path, 'pcb2.jpg'),
    os.path.join(images_inference_path, 'pcb3.jpg'),
    os.path.join(images_inference_path, 'pcb4.jpg'),
    os.path.join(images_inference_path, 'pcb5.jpg'),
    # os.path.join(images_path, '01_short_13_jpg.rf.050e188e3411cb4e85c82831348792a3.jpg'),
    # os.path.join(images_path, '12_open_circuit_08_jpg.rf.3274e85480b532dc5731164617222868.jpg'),
    # os.path.join(images_path, '12_missing_hole_07_jpg.rf.8546bf54fdfa24febc421884764f60e5.jpg'),
]

# image_files_lego = [
#     os.path.join(images_path_lego, 'lego1.JPG'),
# ]

results = model(image_files)

# Process results list
for idx, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    
    # Display result on screen
    result.show()
    
    # Save the result image with index in the filename
    output_filename = f'result_{idx}.jpg'
    result.save(filename=output_filename)
