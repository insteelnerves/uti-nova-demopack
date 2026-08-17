
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

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
        title = "Image"

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage" #configste ne yazıldıysa burada da gerçekleştiği gibi aynı olmalı
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class KeepSideFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class KeepSideTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"

class KeepSideBBox(Config):
    """
        Rotate image without catting off sides.
    """
    name: Literal["KeepSide"] = "KeepSide"
    value: Union[KeepSideTrue, KeepSideFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Keep Sides"

class Degree(Config): # Config class inherit edenler için: name value type field
    """
        Positive angles specify counterclockwise rotation while negative angles indicate clockwise rotation.
    """
    name: Literal["Degree"] = "Degree"
    value: int = Field(ge=-359.0, le=359.0,default=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Angle"

class ExecutionInputs(Inputs): # inputlar yazılır
    inputImage: InputImage

class ExecutionConfigs(Configs): # düğüme tıklanınca sağda gelen ayarlar, hangi sırayla yazdığımız arayüze yansır
    degree: Degree
    drawBBox: KeepSideBBox

class ExecutionRequest(Request): # sistem kodu
    inputs: Optional[ExecutionInputs]
    configs: ExecutionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class ExecutionOutputs(Outputs): # executor un outputları yazılır, birden fazla olabilir
    outputImage: OutputImage
    #outputText: OutputText
    #outputDetection vs.

class ExecutionResponse(Response): # sistem kodu
    outputs: ExecutionOutputs

class Execution(Config): #executor ı tanımlar
    name: Literal["Execution"] = "Execution"
    value: Union[ExecutionRequest, ExecutionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title: "Package"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config): # executorları gösterir
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[Execution] # [A,B,C]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownList"] = "dependentDropdownList"

    class Config:
        title: "Task"
        json_schema_extra = { # bu blok, eğer birden fazla executor varsa, silinmeli
            "target": "value"
        }

class PackageConfigs(Configs): # sistemsel bir kod, değişmez
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component" # capsule component widget
    name: Literal["NovaDemopack"] = "NovaDemopack"
