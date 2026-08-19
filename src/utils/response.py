import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.helper.package import PackageHelper
from components.NovaDemopack.src.models.PackageModel import (
    PackageModel, PackageConfigs, ConfigExecutor,
    TargetDetector, TargetDetectorResponse, TargetDetectorOutputs,
    PoseEstimator, PoseEstimatorResponse, PoseEstimatorOutputs,
    OutputImage, OutputDetections, OutputData
)

def build_target_detector_response(context):
    """Build response model for TargetDetector executor (1 output)."""
    output_detections = OutputDetections(value=context.output_detections)
    outputs = TargetDetectorOutputs(outputDetections=output_detections)

    response = TargetDetectorResponse(outputs=outputs)
    executor = TargetDetector(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )
    return package.build_model(context)


def build_pose_estimator_response(context):
    """Build response model for PoseEstimator executor (2 outputs)."""
    output_image = OutputImage(value=context.output_image)
    output_data = OutputData(value=context.output_data)
    outputs = PoseEstimatorOutputs(outputImage=output_image, outputData=output_data)

    response = PoseEstimatorResponse(outputs=outputs)
    executor = PoseEstimator(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )
    return package.build_model(context)
