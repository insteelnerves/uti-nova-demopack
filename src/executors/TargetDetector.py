"""
TargetDetector executor: Detects targets in an image using OpenCV and filters them.
"""
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.base.model import Detection, BoundingBox

from components.NovaDemopack.src.utils.response import build_target_detector_response
from components.NovaDemopack.src.models.PackageModel import PackageModel


class TargetDetector(Component):
    """
    Executes object detection logic on an input image frame using OpenCV HOG and Haar Cascade.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.input_image = self.request.get_param("inputImage")
        self.filtering_mode = self.request.get_param("ConfigFilteringMode")
        self.output_detections = []
        
        self.hog = self.bootstrap.get('hog')
        self.face_cascade = self.bootstrap.get('face_cascade')

    @staticmethod
    def bootstrap(config: dict) -> dict:
        """
        OpenCV detector'ları bir kez yükler ve tüm frame'ler için bellekte tutar.
        """
        bootstrap_state = {}
        bootstrap_state['hog'] = cv2.HOGDescriptor()
        bootstrap_state['hog'].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        bootstrap_state['face_cascade'] = cv2.CascadeClassifier(cascade_path)
        
        return bootstrap_state

    def _get_filter_params(self):
        """Filtreleme parametrelerini güvenli bir şekilde çeker."""
        mode_val = "standard"
        if isinstance(self.filtering_mode, dict):
            mode_val = self.filtering_mode.get("value", "standard")
        elif self.filtering_mode:
            mode_val = self.filtering_mode

        params = {"mode": mode_val, "threshold": 0.5, "classes": []}

        if mode_val == "standard":
            threshold = self.request.get_param("ConfidenceThreshold")
            if threshold is None or isinstance(threshold, dict):
                threshold = 0.5
            params["threshold"] = float(threshold)
            
        elif mode_val == "classFilter":
            selected_classes = self.request.get_param("TargetClasses") or []
            if isinstance(selected_classes, dict):
                selected_classes = selected_classes.get("value", [])
            params["classes"] = selected_classes
            
        return params

    def process_detection(self, cv_img):
        """Gerçek görüntü üzerinde OpenCV ile nesne tespiti yapar ve filtreler."""
        h, w = cv_img.shape[:2]
        self.output_detections = []
        params = self._get_filter_params()
        
        if self.hog is not None:
            boxes, weights = self.hog.detectMultiScale(
                cv_img, 
                winStride=(8, 8), 
                padding=(16, 16), 
                scale=1.05
            )
            
            for (x, y, box_w, box_h), weight in zip(boxes, weights):
                norm_left = x / w
                norm_top = y / h
                norm_width = box_w / w
                norm_height = box_h / h
                
                detection = Detection(
                    boundingBox=BoundingBox(
                        left=norm_left,
                        top=norm_top,
                        width=norm_width,
                        height=norm_height
                    ),
                    confidence=float(weight),
                    classLabel="person",
                    classId=0
                )
                self.output_detections.append(detection)
        
        if self.face_cascade is not None:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, box_w, box_h) in faces:
                norm_left = x / w
                norm_top = y / h
                norm_width = box_w / w
                norm_height = box_h / h
                
                detection = Detection(
                    boundingBox=BoundingBox(
                        left=norm_left,
                        top=norm_top,
                        width=norm_width,
                        height=norm_height
                    ),
                    confidence=0.9, 
                    classLabel="face",
                    classId=1
                )
                self.output_detections.append(detection)
        
        if params["mode"] == "standard":
            self.output_detections = [
                d for d in self.output_detections if d.confidence >= params["threshold"]
            ]
        elif params["mode"] == "classFilter" and params["classes"]:
            self.output_detections = [
                d for d in self.output_detections if d.classLabel in params["classes"]
            ]

    def run(self):
        img_obj = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
        
        if img_obj is not None and hasattr(img_obj, 'value'):
            cv_img = img_obj.value
            if cv_img.dtype != np.uint8:
                cv_img = cv_img.astype(np.uint8)
            
            self.process_detection(cv_img)
        else:
            self.output_detections = []

        return build_target_detector_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
