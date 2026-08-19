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

    def _extract_detection_info(self, detection):
        """
        SDK veriyi dict olarak gönderebileceği için, hem Pydantic objesini 
        hem de Dictionary'yi güvenli bir şekilde işleyen yardımcı metod.
        """
        if detection is None:
            return None, 0.0, "unknown"

        # 1. Bounding Box'ı çek (Objeden veya Dict'ten)
        bbox = None
        if hasattr(detection, 'boundingBox'):
            bbox = detection.boundingBox
        elif isinstance(detection, dict):
            bbox = detection.get('boundingBox')

        if bbox is None:
            return None, 0.0, "unknown"

        # 2. Koordinatları çek
        if hasattr(bbox, 'left'):
            left, top, width, height = bbox.left, bbox.top, bbox.width, bbox.height
        elif isinstance(bbox, dict):
            left = bbox.get('left', 0)
            top = bbox.get('top', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
        else:
            return None, 0.0, "unknown"

        # 3. Confidence ve Class Label'ı çek
        if hasattr(detection, 'confidence'):
            conf = detection.confidence
            cls_label = getattr(detection, 'classLabel', 'unknown')
        elif isinstance(detection, dict):
            conf = detection.get('confidence', 0.0)
            cls_label = detection.get('classLabel', 'unknown')
        else:
            conf = 0.0
            cls_label = "unknown"

        return {'left': left, 'top': top, 'width': width, 'height': height}, float(conf), cls_label

    def calculate_pose(self, detection_info):
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

        bbox, confidence, class_label = detection_info

        # Tespit varsa merkezini hesapla, yoksa görselin ortasını al
        if bbox:
            center_x = bbox['left'] + bbox['width'] / 2
            center_y = bbox['top'] + bbox['height'] / 2
        else:
            center_x, center_y = 0.5, 0.5

        self.output_data = {
            "target_id": 1,
            "class_label": class_label,
            "coordinates": {
                "x": round(center_x, 4),
                "y": round(center_y, 4),
                "z": round(distance, 2)
            },
            "unit": "meters",
            "confidence": confidence
        }
        return distance

    def draw_pose_info(self, cv_img, distance, detection_info):
        """Draws pose overlay on image - uses detection bbox if available."""
        h, w = cv_img.shape[:2]
        bbox, confidence, class_label = detection_info
        
        if bbox:
            # Gerçek tespit kutusunu piksele çevir
            left = int(bbox['left'] * w)
            top = int(bbox['top'] * h)
            right = int((bbox['left'] + bbox['width']) * w)
            bottom = int((bbox['top'] + bbox['height']) * h)
            cx = (left + right) // 2
            cy = (top + bottom) // 2
        else:
            # Fallback: Görselin ortası
            cx, cy = w // 2, h // 2
            box_w, box_h = int(w * 0.4), int(h * 0.4)
            left = cx - box_w // 2
            top = cy - box_h // 2
            right = left + box_w
            bottom = top + box_h

        # Çizimleri yap
        cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.drawMarker(cv_img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        
        label = f"{class_label.upper()} | Z: {distance:.2f}m"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(cv_img, (left, top - label_h - 10), (left + label_w + 10, top), (0, 0, 0), -1)
        cv2.putText(cv_img, label, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        return cv_img

    def run(self):
        # Görüntüyü Redis'ten çek
        img_obj = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
        
        # inputDetections listesi veya tekil dict olabilir, ilkini al
        detection = None
        if self.input_detections:
            if isinstance(self.input_detections, list) and len(self.input_detections) > 0:
                detection = self.input_detections[0]
            elif isinstance(self.input_detections, dict):
                detection = self.input_detections

        # Hem dict hem obje uyumlu bilgi çıkarma
        detection_info = self._extract_detection_info(detection)
        
        # Mesafe hesapla
        distance = self.calculate_pose(detection_info)

        # Görüntü üzerine çizim yap
        if img_obj is not None and hasattr(img_obj, 'value'):
            cv_img = img_obj.value
            if cv_img.dtype != np.uint8:
                cv_img = cv_img.astype(np.uint8)
            
            self.draw_pose_info(cv_img, distance, detection_info)
            img_obj.value = cv_img

        # İşlenen görüntüyü Redis'e yaz
        self.output_image = Image.set_frame(
            img=img_obj,
            package_uID=self.uID,
            redis_db=self.redis_db
        )
        
        return build_pose_estimator_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
