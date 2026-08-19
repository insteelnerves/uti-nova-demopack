"""
    TargetDetector executor: Detects targets in an image and filters them.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.base.model import Detection, BoundingBox
from components.NovaDemopack.src.utils.response import build_target_detector_response
from components.NovaDemopack.src.models.PackageModel import PackageModel


class TargetDetector(Component):
    """
    Executes object detection logic on an input image frame.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.input_image = self.request.get_param("inputImage")

        self.filtering_mode = self.request.get_param("ConfigFilteringMode")

        self.output_detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def process_detection(self):
        """Simulate target detection filtering based on configuration."""
        sample_detection = Detection(
            boundingBox=BoundingBox(left=0.2, top=0.2, width=0.3, height=0.3),
            confidence=0.85,
            classLabel="target",
            classId=0
        )

        # Apply filtering logic according to dependent dropdown choice
        if self.filtering_mode == "standard":
            threshold = self.request.get_param("ConfidenceThreshold") or 0.5
            if sample_detection.confidence >= threshold:
                self.output_detections.append(sample_detection)
        elif self.filtering_mode == "classFilter":
            selected_classes = self.request.get_param("TargetClasses") or []
            if sample_detection.classLabel in selected_classes:
                self.output_detections.append(sample_detection)

    def run(self):
        _ = Image.get_frame(img=self.input_image, redis_db=self.redis_db)

        self.process_detection()

        return build_target_detector_response(context=self)

if "__main__" == __name__:
    Executor(sys.argv[1]).run()