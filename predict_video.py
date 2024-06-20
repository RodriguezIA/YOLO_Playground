# from ultralytics import YOLO
# # from ultralytics.solutions import object_counter
# import cv2

# model = YOLO("yolov8m.pt")
# cap = cv2.VideoCapture("videos\city1.mp4")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Define region points
# # region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# # Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi",
#                        cv2.VideoWriter_fourcc(*'mp4v'),
#                        fps,
#                        (w, h))

# # Init Object Counter
# # counter = object_counter.ObjectCounter()
# # counter.set_args(view_img=True,
# #                  reg_pts=region_points,
# #                  classes_names=model.names,
# #                  draw_tracks=True,
# #                  line_thickness=2)

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     # tracks = model.track(im0, persist=True, show=False)

#     # im0 = counter.start_counting(im0, tracks)
#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# se instalo supervision
# import supervision as sv
# import numpy as np
# from ultralytics import YOLO

# video_path = 'videos/city1.mp4'
# model = YOLO('yolov8m.pt')

# video_info = sv.VideoInfo.from_video_path(video_path)

# def process_frame(frame: np.ndarray, _) -> np.ndarray:
#     results = model(frame, imgsz=1280)[0]
    
#     detections = sv.Detections.from_yolov8(results)

#     box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

#     labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
#     frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

#     return frame

# sv.process_video(source_path=video_path, target_path=f"result.mp4", callback=process_frame)

# chat fix
import supervision as sv
import numpy as np
from ultralytics import YOLO

video_path = 'videos/IMG_0135.mp4'
model = YOLO('best.pt')

video_info = sv.VideoInfo.from_video_path(video_path)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]

    # Manually create detections from YOLOv8 results
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else np.empty((0, 4))
    confidences = results.boxes.conf.cpu().numpy() if results.boxes else np.empty((0,))
    class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else np.empty((0,), dtype=int)

    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=class_ids,
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)

    labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

sv.process_video(source_path=video_path, target_path="result.mp4", callback=process_frame)
