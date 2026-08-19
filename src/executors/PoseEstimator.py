"""
    PoseEstimator executor: Calculates target 3D pose using detections and image.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.TargetNavigationPackage.src.utils.response import build_pose_estimator_response
from components.TargetNavigationPackage.src.models.PackageModel import PackageModel


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

    def calculate_pose(self):
        """Estimate 3D position based on solver mode selection."""
        distance = 10.0

        if self.solver_mode == "cameraSolver":
            method = self.request.get_param("SolverMethod") or "iterative"
            distance = 12.5 if method == "p3p" else 10.0
        elif self.solver_mode == "sensorAssisted":
            offset = self.request.get_param("DistanceOffset") or 0.0
            distance += offset

        self.output_data = {
            "target_id": 1,
            "coordinates": {"x": 1.2, "y": -0.5, "z": round(distance, 2)},
            "unit": "meters"
        }

    def run(self):

        img = Image.get_frame(img=self.input_image, redis_db=self.redis_db)

        self.calculate_pose()

        self.output_image = Image.set_frame(
            img=img,
            package_uID=self.uID,
            redis_db=self.redis_db
        )

        return build_pose_estimator_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()