import rclpy
from rclpy.node import Node
import cv2
import json
import numpy as np
import onnxruntime as ort
import argparse
import os
import matplotlib.pyplot as plt
import threading
import queue
import time
import platform

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

frame_queue = queue.Queue(maxsize=3) # A queue to hold frames

def frame_capture_thread(cap, is_running):
    try:
        os.nice(-10)
    except:
        pass
    frame_count = 0
    start_time = time.time()
    read_times = []
    while is_running.is_set():
        read_start = time.time()
        ret, frame = cap.read()
        read_times.append(time.time() - read_start)
        if not ret:
            time.sleep(0.01)
            continue
        try:
            frame_queue.put(frame, timeout=0.1)
            frame_count += 1
        except queue.Full:
            pass # Drop frame if the main thread is lagging
        if frame_count % 120 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"Frame Reception Rate: {fps:.2f} FPS; Avg. cap.read(): {sum(read_times) / len(read_times) * 1000:.2f}ms")
            # Reset counters
            frame_count = 0
            start_time = time.time()
            read_times = []

def xywh2xyxy(box):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    coord = np.copy(box)
    coord[..., 0] = box[..., 0] - box[..., 2] / 2  # x1
    coord[..., 1] = box[..., 1] - box[..., 3] / 2  # y1
    coord[..., 2] = box[..., 0] + box[..., 2] / 2  # x2
    coord[..., 3] = box[..., 1] + box[..., 3] / 2  # y2
    return coord

class YoloInferenceNode(Node):
    def __init__(self, headless, hitl, hfov, vfov):
        super().__init__('yolo_inference_node')
        self.headless = headless
        self.hitl = hitl
        self.hfov = hfov
        self.vfov = vfov
        self.architecture = platform.machine()
        
        # Load classes
        names_file = "/aas/yolo/coco.json"
        with open(names_file, "r") as f:
            self.classes = {int(k): v for k, v in json.load(f).items()}
        colors_rgba = plt.cm.hsv(np.linspace(0, 1, len(self.classes)))
        self.colors = (colors_rgba[:, [2, 1, 0]] * 255).astype(np.uint8) # From RGBA (0-1 float) to BGR (0-255 int)

        # Load model runtime
        model_path = "/aas/yolo/yolov8n.onnx" # Model options (from fastest to most accurate, <10MB to >100MB): yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        if self.architecture == 'x86_64':
            print("Loading CUDAExecutionProvider on AMD64 (x86) architecture.")
            self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"]) # For simulation
        elif self.architecture == 'aarch64':
            print("Loading (with cache) TensorrtExecutionProvider on ARM64 architecture (Jetson).") # The first cache built takes ~3'
            cache_path = "/tensorrt_cache" # Mounted as volume by main_deploy.sh
            os.makedirs(cache_path, exist_ok=True)
            provider_options = {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_path,
            }
            self.session = ort.InferenceSession(
                model_path,
                providers=[('TensorrtExecutionProvider', provider_options)] # For deployment on Jetson Orin
            )
        else:
            print(f"Loading CPUExecutionProvider on an unknown architecture: {self.architecture}")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]) # Backup, not recommended
        self.input_name = self.session.get_inputs()[0].name
        
        # Confirm execution providers
        self.get_logger().info(f"Execution providers in use: {self.session.get_providers()}")
        
        # Create publishers
        self.detection_publisher = self.create_publisher(Detection2DArray, 'detections', 10)
        # self.image_publisher = self.create_publisher(Image, 'detections_image', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("YOLO inference started.")

    def ros_spin_thread(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001) # This is only to get the simulation time from /clock

    def run_inference_loop(self):
        # Acquire video stream
        if self.architecture == 'x86_64':
            # # GPU pipeline: NOT WORKING TODO: switch to base image with DeepStream
            # gst_pipeline_string = (
            #     "udpsrc port=5600 ! "
            #     "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
            #     "rtph264depay ! "
            #     "h264parse ! "
            #     "nvh264dec ! "
            #     "nvvidconv ! "  # Use NVIDIA's GPU-accelerated converter
            #     "video/x-raw(memory:NVMM), format=BGRx ! "
            #     "videoconvert ! "
            #     "video/x-raw, format=BGR ! appsink"
            # )
            # CPU pipeline
            gst_pipeline_string = (
                "udpsrc port=5600 ! "
                "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                "rtph264depay ! "
                "avdec_h264 threads=4 ! " # Use CPU decoder, threads=0 for autodetection
                "videoconvert ! "
                "video/x-raw, format=BGR ! appsink"
            )
            cap = cv2.VideoCapture(gst_pipeline_string, cv2.CAP_GSTREAMER)
        elif self.architecture == 'aarch64':
            if self.hitl: # For HITL, acquire UDP stream from gz-sim
                # GPU pipeline:
                gst_pipeline_string = (
                "udpsrc port=5600 ! "
                    "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                    "rtph264depay ! "
                    "h264parse ! "
                    "nvv4l2decoder ! "     # Hardware Decoding: Uses the Orin's dedicated engine
                    "nvvidconv ! "         # NVMM-to-CPU Memory Conversion
                    "video/x-raw, format=I420 ! "
                    "videoconvert ! "      # CPU Color Conversion: I420 to BGR
                    "video/x-raw, format=BGR ! "
                    "appsink drop=true max-buffers=1 "
                )
                # # CPU pipeline:
                # gst_pipeline_string = (
                #     "udpsrc port=5600 ! "
                #     "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                #     "rtph264depay ! "
                #     "avdec_h264 ! "      # Generic CPU H.264 decoder
                #     "videoconvert ! "
                #     "video/x-raw, format=BGR ! appsink"
                # )
                cap = cv2.VideoCapture(gst_pipeline_string, cv2.CAP_GSTREAMER)
            else: # Default, acquire CSI camera 
                # GPU pipeline:
                gst_pipeline_string = (
                    "nvarguscamerasrc sensor-id=0 ! "
                    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
                    "nvvidconv ! "
                    "video/x-raw, format=BGRx, width=1280, height=720, framerate=60/1 ! "
                    "videoconvert ! "
                    "appsink drop=true max-buffers=1 sync=false"
                ) # Test with: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1' ! nvvidconv ! nv3dsink -e
                cap = cv2.VideoCapture(gst_pipeline_string, cv2.CAP_GSTREAMER)
        # cap = cv2.VideoCapture("/sample.mp4") # Load sample video for testing
        assert cap.isOpened(), "Failed to open video stream"
        print(f"Pipeline FPS: {cap.get(cv2.CAP_PROP_FPS)}")

        if not self.headless:
            drone_id = os.getenv('DRONE_ID', '0')
            self.WINDOW_NAME = f"YOLOv8 (Aircraft {drone_id})"
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.WINDOW_NAME, 800+(int(drone_id)-1)*25, 5+(int(drone_id)-1)*200)
            # cv2.resizeWindow(self.WINDOW_NAME, 400, 200)

        # Start the video capture thread
        is_running = threading.Event()
        is_running.set()
        frame_thread = threading.Thread(target=frame_capture_thread, args=(cap, is_running), daemon=True)
        frame_thread.start()

        # ROS spinning thread
        ros_thread = threading.Thread(target=self.ros_spin_thread, daemon=True)
        ros_thread.start()

        inference_count = 0
        start_time = time.time()
        yolo_times = []
        self.session_times = []

        while rclpy.ok():
            try:
                frame = frame_queue.get(timeout=1) # Get the most recent frame from the queue
            except queue.Empty:
                self.get_logger().info("Frame queue is empty, is the stream running?")
                continue
            
            # Inference
            yolo_start = time.time()
            boxes, confidences, class_ids = self.run_yolo(frame)
            yolo_times.append(time.time() - yolo_start)

            inference_count += 1
            if inference_count % 120 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                yolo_fps = inference_count / elapsed_time
                print(f"YOLO Inference Rate: {yolo_fps:.2f} FPS; Avg. YOLO: {sum(yolo_times) / len(yolo_times)*1000:.2f}ms; Avg. session: {sum(self.session_times) / len(self.session_times)*1000:.2f}ms")
                # Reset counters
                inference_count = 0
                start_time = time.time()
                yolo_times = []
                self.session_times = []

            # Publish detections
            self.publish_detections(frame, boxes, confidences, class_ids)

            # Visualize
            if not self.headless:
                self.visualize(frame, boxes, confidences, class_ids)
                cv2.waitKey(1)

        # Cleanup
        is_running.clear()
        frame_thread.join()
        
        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()

    def run_yolo(self, frame):
        h0, w0 = frame.shape[:2]
        INPUT_SIZE = 640 # YOLOv8 input size
        
        img = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        
        session_start = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        self.session_times.append(time.time() - session_start)
        
        preds = outputs[0][0].T
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        confidences = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        
        # Filter
        CONF_THRESH = 0.5
        mask = confidences > CONF_THRESH
        
        if not mask.any():
            return np.array([]), np.array([]), np.array([])
        
        # Apply mask once
        boxes_filtered = boxes[mask]
        confidences_filtered = confidences[mask]
        class_ids_filtered = class_ids[mask]
        
        # Convert to xyxy once
        boxes_xyxy = xywh2xyxy(boxes_filtered)
        
        # Apply Non-Maximal Suppression
        NMS_THRESH = 0.45
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences_filtered.tolist(), CONF_THRESH, NMS_THRESH)
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Scale 
        final_boxes = boxes_xyxy[indices]
        scale = np.array([w0 / INPUT_SIZE, h0 / INPUT_SIZE, w0 / INPUT_SIZE, h0 / INPUT_SIZE])
        final_boxes *= scale
        
        return final_boxes, confidences_filtered[indices], class_ids_filtered[indices]

    def publish_detections(self, frame, boxes, confidences, class_ids):
        detection_array_msg = Detection2DArray()
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        detection_array_msg.header.frame_id = "camera_frame"
        image_height, image_width = frame.shape[:2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            bbox = BoundingBox2D()
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            bbox.center.position.x = center_x
            bbox.center.position.y = center_y
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)

            offset_x = center_x - (image_width / 2.0)
            offset_y = (image_height / 2.0) - center_y
            norm_x = offset_x / image_width
            norm_y = offset_y / image_height
            azimuth = norm_x * self.hfov
            elevation = norm_y * self.vfov

            hypothesis = ObjectHypothesis()
            hypothesis.class_id = str(self.classes[class_ids[i]])
            hypothesis.score = float(confidences[i])

            result = ObjectHypothesisWithPose()
            result.hypothesis = hypothesis
            result.pose.pose.position.x = azimuth # degrees
            result.pose.pose.position.y = elevation # degrees
            
            detection = Detection2D() 
            detection.bbox = bbox
            detection.id = str(self.classes[class_ids[i]]) 
            detection.results.append(result)
            detection_array_msg.detections.append(detection)

        self.detection_publisher.publish(detection_array_msg)
        
        # if not self.headless:
        #     self.image_publisher.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def visualize(self, frame, boxes, confidences, class_ids):
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = confidences[i]
            class_id = class_ids[i]
            class_name = self.classes[class_id]
            color = tuple(self.colors[class_id, [2, 1, 0]].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.imshow(self.WINDOW_NAME, frame)

def main(args=None):
    parser = argparse.ArgumentParser(description="YOLOv8 ROS2 Inference Node.")
    parser.add_argument('--headless', action='store_true', help="Run in headless mode.")
    parser.add_argument('--hitl', action='store_true', help="Open camerafrom gz-sim for HITL.")
    parser.add_argument('--hfov', type=float, default=90.0, help="Horizontal field of view in degrees.")
    parser.add_argument('--vfov', type=float, default=60.0, help="Vertical field of view in degrees.")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    yolo_node = YoloInferenceNode(headless=cli_args.headless, hitl=cli_args.hitl, hfov=cli_args.hfov, vfov=cli_args.vfov)
    yolo_node.run_inference_loop()
    
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
