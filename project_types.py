from typing import Dict, List, TypedDict
from torch import Tensor

StateDict = Dict[str, Tensor]

class Transaction(TypedDict):
    tx_id: str
    approved_tx_ids: List[str]
    state_dict: StateDict
    creator_id: str
    timestamp: str
