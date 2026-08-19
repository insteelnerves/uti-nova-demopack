from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Image, Detection, Inputs, Configs, Outputs,
    Response, Request, Output, Input, Config
)

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Input Image"

class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], Detection]
    type: str = "object"

    class Config:
        title = "Input Detections"

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Output Image"

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: Union[List[Detection], Detection]
    type: str = "object"

    class Config:
        title = "Output Detections"

class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[dict, list]
    type: str = "object"

    class Config:
        title = "Target Pose Data"

class ConfidenceThreshold(Config):
    """Sets the minimum confidence score required to keep a detection."""
    name: Literal["ConfidenceThreshold"] = "ConfidenceThreshold"
    value: float = Field(ge=0.0, le=1.0, default=0.5)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Confidence Threshold"
        json_schema_extra = {"shortDescription": "Min Confidence Score"}

class ClassOptionTarget(Config):
    name: Literal["target"] = "target"
    value: Literal["target"] = "target"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Target"

class ClassOptionLandingZone(Config):
    name: Literal["landingZone"] = "landingZone"
    value: Literal["landingZone"] = "landingZone"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Landing Zone"

class TargetClasses(Config):
    """Select target classes to filter detections."""
    name: Literal["TargetClasses"] = "TargetClasses"
    value: List[Union[ClassOptionTarget, ClassOptionLandingZone]]
    type: Literal["object"] = "object"
    field: Literal["selectBox"] = "selectBox"

    class Config:
        title = "Filter Classes"
        json_schema_extra = {"shortDescription": "Selected Target Classes"}

class StandardModeOption(Config):
    name: Literal["StandardModeOption"] = "StandardModeOption"
    confidenceThreshold: ConfidenceThreshold
    value: Literal["standard"] = "standard"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Standard Threshold Mode"

class ClassFilterModeOption(Config):
    name: Literal["ClassFilterModeOption"] = "ClassFilterModeOption"
    targetClasses: TargetClasses
    value: Literal["classFilter"] = "classFilter"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Class Filter Mode"

class ConfigFilteringMode(Config):
    """Select detection filtering strategy."""
    name: Literal["ConfigFilteringMode"] = "ConfigFilteringMode"
    value: Union[StandardModeOption, ClassFilterModeOption]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Filtering Mode"
        json_schema_extra = {"shortDescription": "Detection Filter Strategy"}

class OptionIterative(Config):
    name: Literal["iterative"] = "iterative"
    value: Literal["iterative"] = "iterative"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "SOLVEPNP_ITERATIVE"

class OptionP3P(Config):
    name: Literal["p3p"] = "p3p"
    value: Literal["p3p"] = "p3p"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "SOLVEPNP_P3P"

class SolverMethod(Config):
    """Select solvePnP estimation algorithm."""
    name: Literal["SolverMethod"] = "SolverMethod"
    value: Union[OptionIterative, OptionP3P]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Solver Method"
        json_schema_extra = {"shortDescription": "OpenCV solvePnP Method"}

class DistanceOffset(Config):
    """Offset distance in meters for sensor calibration."""
    name: Literal["DistanceOffset"] = "DistanceOffset"
    value: float = Field(default=0.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["0.0"] = "0.0"

    class Config:
        title = "Distance Offset (m)"
        json_schema_extra = {"shortDescription": "Sensor Offset in Meters"}

class CameraSolverOption(Config):
    name: Literal["CameraSolverOption"] = "CameraSolverOption"
    solverMethod: SolverMethod
    value: Literal["cameraSolver"] = "cameraSolver"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Camera Vision Mode"

class SensorAssistedOption(Config):
    name: Literal["SensorAssistedOption"] = "SensorAssistedOption"
    distanceOffset: DistanceOffset
    value: Literal["sensorAssisted"] = "sensorAssisted"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Sensor-Assisted Mode"

class ConfigSolverMode(Config):
    """Select pose estimation calculation mode."""
    name: Literal["ConfigSolverMode"] = "ConfigSolverMode"
    value: Union[CameraSolverOption, SensorAssistedOption]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Solver Mode"
        json_schema_extra = {"shortDescription": "Pose Estimation Algorithm Mode"}

class TargetDetectorInputs(Inputs):
    inputImage: InputImage

class TargetDetectorConfigs(Configs):
    configFilteringMode: ConfigFilteringMode

class TargetDetectorOutputs(Outputs):
    outputDetections: OutputDetections

class TargetDetectorRequest(Request):
    inputs: Optional[TargetDetectorInputs]
    configs: TargetDetectorConfigs

    class Config:
        json_schema_extra = {"target": "configs"}

class TargetDetectorResponse(Response):
    outputs: TargetDetectorOutputs

class TargetDetector(Config):
    name: Literal["TargetDetector"] = "TargetDetector"
    value: Union[TargetDetectorRequest, TargetDetectorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Target Detector"
        json_schema_extra = {"target": {"value": 0}}

class PoseEstimatorInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections

class PoseEstimatorConfigs(Configs):
    configSolverMode: ConfigSolverMode

class PoseEstimatorOutputs(Outputs):
    outputImage: OutputImage
    outputData: OutputData

class PoseEstimatorRequest(Request):
    inputs: Optional[PoseEstimatorInputs]
    configs: PoseEstimatorConfigs

    class Config:
        json_schema_extra = {"target": "configs"}

class PoseEstimatorResponse(Response):
    outputs: PoseEstimatorOutputs

class PoseEstimator(Config):
    name: Literal["PoseEstimator"] = "PoseEstimator"
    value: Union[PoseEstimatorRequest, PoseEstimatorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Pose Estimator"
        json_schema_extra = {"target": {"value": 0}}

class ConfigExecutor(Config):
    """Select task executor."""
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[TargetDetector, PoseEstimator]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Select Package Operation"}

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["NovaDemopack"] = "NovaDemopack"
