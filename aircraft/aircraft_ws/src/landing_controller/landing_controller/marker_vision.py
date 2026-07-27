import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class MarkerObservation:
    detected: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    seen_at: float = 0.0
    corners: Optional[np.ndarray] = None


class MarkerTracker:
    def __init__(self, marker_id: int, min_area_px: float):
        self.marker_id = marker_id
        self.min_area_px = min_area_px
        self.detector = self._create_detector()

    def detect(self, frame: np.ndarray) -> MarkerObservation:
        corners, ids = self._detect(frame)
        if ids is None:
            return MarkerObservation()

        matches = np.where(ids.flatten() == self.marker_id)[0]
        if len(matches) == 0:
            return MarkerObservation()

        pts = corners[int(matches[0])][0]
        area = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
        if area < self.min_area_px:
            return MarkerObservation()

        center = pts.mean(axis=0)
        return MarkerObservation(
            detected=True,
            center_x=float(center[0]),
            center_y=float(center[1]),
            area=area,
            seen_at=time.time(),
            corners=pts,
        )

    def draw(self, frame: np.ndarray, marker: MarkerObservation, state: str, command_text: str) -> np.ndarray:
        height, width = frame.shape[:2]
        image_center = (width // 2, height // 2)
        cv2.drawMarker(frame, image_center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=24)

        if marker.detected and marker.corners is not None:
            cv2.polylines(frame, [marker.corners.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(marker.center_x), int(marker.center_y)), 4, (255, 0, 0), -1)

        cv2.putText(frame, state, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, command_text, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def _detect(self, frame: np.ndarray):
        if hasattr(cv2.aruco, 'ArucoDetector') and not isinstance(self.detector, tuple):
            corners, ids, _ = self.detector.detectMarkers(frame)
            return corners, ids

        aruco_dict, params = self.detector
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
        return corners, ids

    def _create_detector(self):
        aruco_dict = (
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            if hasattr(cv2.aruco, 'getPredefinedDictionary')
            else cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        )
        params = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, 'DetectorParameters_create')
            else cv2.aruco.DetectorParameters()
        )
        if hasattr(cv2.aruco, 'ArucoDetector'):
            return cv2.aruco.ArucoDetector(aruco_dict, params)
        return aruco_dict, params


class DebugVideo:
    def __init__(self, enabled: bool, window_name: str, logger):
        self.enabled = enabled
        self.window_name = window_name
        self.logger = logger
        self.opened = False

    def open(self) -> bool:
        if not self.enabled:
            return False
        if platform.system() == 'Linux' and not (os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')):
            self.logger.warn('Video disabled: DISPLAY/WAYLAND_DISPLAY is not set.')
            return False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            self.logger.warn(f'Video disabled: {exc}')
            return False
        self.opened = True
        return True

    def show(self, frame: np.ndarray):
        if not self.opened:
            return
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def close(self):
        if not self.opened:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self.opened = False


class GStreamerCamera:
    def __init__(self, port: int, logger):
        self.port = port
        self.logger = logger
        self.capture = None
        self.running = False
        self.thread = None

    def start(self, on_frame: Callable[[np.ndarray], None]):
        pipeline = self._pipeline()
        if pipeline is None:
            return

        self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            self.logger.warn('GStreamer camera stream unavailable; waiting for ROS camera topic.')
            self.capture.release()
            self.capture = None
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(on_frame,), daemon=True)
        self.thread.start()
        self.logger.info('Opened GStreamer camera stream.')

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _loop(self, on_frame: Callable[[np.ndarray], None]):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                on_frame(frame)
            else:
                time.sleep(0.05)

    def _pipeline(self) -> Optional[str]:
        machine = platform.machine()
        if machine == 'x86_64':
            return (
                f'udpsrc port={self.port} ! '
                'application/x-rtp, media=(string)video, encoding-name=(string)H264 ! '
                'rtph264depay ! avdec_h264 ! videoconvert ! '
                'video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false'
            )

        if machine == 'aarch64' and os.getenv('HITL', 'false').lower() == 'true':
            return (
                f'udpsrc port={self.port} ! '
                'application/x-rtp, media=(string)video, encoding-name=(string)H264 ! '
                'rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! '
                'video/x-raw, format=I420 ! videoconvert ! video/x-raw, format=BGR ! '
                'appsink drop=true max-buffers=1'
            )

        return None
