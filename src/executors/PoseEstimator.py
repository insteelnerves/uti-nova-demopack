"""
PoseEstimator executor: Calculates target 3D pose using detections and image.
"""
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.NovaDemopack.src.utils.response import build_pose_estimator_response
from components.NovaDemopack.src.models.PackageModel import PackageModel


class PoseEstimator(Component):
    """
    Executes target 3D pose estimation using vision camera solver or sensor assistance.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.input_image = self.request.get_param("inputImage")
        self.input_detections = self.request.get_param("inputDetections")
        self.solver_mode = self.request.get_param("ConfigSolverMode")
        self.output_image = None
        self.output_data = {}

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def calculate_pose(self, detection=None):
        """Estimate 3D position based on solver mode selection."""
        distance = 10.0
        
        if self.solver_mode == "cameraSolver":
            method = self.request.get_param("SolverMethod") or "iterative"
            if isinstance(method, dict):
                method = method.get("value", "iterative")
            distance = 12.5 if method == "p3p" else 10.0
        elif self.solver_mode == "sensorAssisted":
            offset = self.request.get_param("DistanceOffset")
            if offset is None or isinstance(offset, dict):
                offset = 0.0
            distance += float(offset)

        # Use detection center if available
        if detection and hasattr(detection, 'boundingBox') and detection.boundingBox:
            bbox = detection.boundingBox
            center_x = bbox.left + bbox.width / 2
            center_y = bbox.top + bbox.height / 2
        else:
            center_x, center_y = 0.5, 0.5

        self.output_data = {
            "target_id": 1,
            "coordinates": {
                "x": round(center_x, 4),
                "y": round(center_y, 4),
                "z": round(distance, 2)
            },
            "unit": "meters",
            "confidence": detection.confidence if detection else 0.0
        }
        return distance

    def draw_pose_info(self, cv_img, distance, detection=None):
        """Draws pose overlay on image - uses detection bbox if available."""
        h, w = cv_img.shape[:2]
        
        if detection and hasattr(detection, 'boundingBox') and detection.boundingBox:
            # Use real detection bounding box
            bbox = detection.boundingBox
            left = int(bbox.left * w)
            top = int(bbox.top * h)
            right = int((bbox.left + bbox.width) * w)
            bottom = int((bbox.top + bbox.height) * h)
            cx = (left + right) // 2
            cy = (top + bottom) // 2
        else:
            # Fallback: center of image
            cx, cy = w // 2, h // 2
            box_w, box_h = int(w * 0.4), int(h * 0.4)
            left = cx - box_w // 2
            top = cy - box_h // 2
            right = left + box_w
            bottom = top + box_h

        # Draw bounding box (red)
        cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 0, 255), 3)
        
        # Draw center cross (cyan)
        cv2.drawMarker(cv_img, (cx, cy), (0, 255, 255),
                       cv2.MARKER_CROSS, 40, 3)
        
        # Draw label background (black)
        label = f"TARGET | Z: {distance:.2f}m"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(cv_img, 
                      (left, top - label_h - 10), 
                      (left + label_w + 10, top), 
                      (0, 0, 0), -1)
        
        # Draw label text (green)
        cv2.putText(cv_img, label, (left + 5, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        return cv_img

    def run(self):
        # Get image from Redis
        img_obj = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
        
        # Get first detection if available
        detection = None
        if self.input_detections:
            if isinstance(self.input_detections, list) and len(self.input_detections) > 0:
                detection = self.input_detections[0]
            elif hasattr(self.input_detections, 'boundingBox'):
                detection = self.input_detections
        
        # Calculate pose
        distance = self.calculate_pose(detection)

        # Process image
        if img_obj is not None and hasattr(img_obj, 'value'):
            cv_img = img_obj.value
            
            # Convert to uint8 if needed (cv2 requires uint8)
            if cv_img.dtype != np.uint8:
                cv_img = cv_img.astype(np.uint8)
            
            # Draw on image (inplace)
            self.draw_pose_info(cv_img, distance, detection)
            
            # Update image value
            img_obj.value = cv_img

        # Store to Redis
        self.output_image = Image.set_frame(
            img=img_obj,
            package_uID=self.uID,
            redis_db=self.redis_db
        )
        
        return build_pose_estimator_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
