from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses_json import dataclass_json
from PIL import Image
from torch import Tensor


@dataclass_json
@dataclass
class BaseRequest:
    type: str


@dataclass_json
@dataclass
class StartSessionRequest(BaseRequest):
    path: str
    session_id: Optional[str] = None


@dataclass_json
@dataclass
class CloseSessionRequest(BaseRequest):
    session_id: str


@dataclass_json
@dataclass
class TextPromptRequest(BaseRequest):
    session_id: str
    prompt_text: str


@dataclass_json
@dataclass
class OpenSessionResponse:
    session_id: str



@dataclass_json
@dataclass
class CloseSessionResponse:
    success: bool


@dataclass_json
@dataclass
class TextPromptResponse:
    response_text: str


@dataclass_json
@dataclass
class InferenceSession:
    start_time: float
    last_use_time: float
    session_id: str
    state: Dict[str, Tensor]
