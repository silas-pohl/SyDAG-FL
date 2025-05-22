from __future__ import annotations
import threading
import torch
import networkx as nx
from datetime import datetime
from typing import TYPE_CHECKING

from cnn import CNN
from save_load_utils import save_state_dict_to_disk

if TYPE_CHECKING:
    from workers import AbstractWorker
    from femnist_dataset_manager import FEMNISTDatasetManager
    from project_types import Transaction

class FLSimulation():

    id: str
    approach: dict
    attack_scenario: dict|None
    concurrent_honest_workers: int
    evaluation_interval: int
    stop_threshold: int
    dataset_manager: FEMNISTDatasetManager

    device: torch.device                    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tangle: nx.DiGraph                      = nx.DiGraph()
    tangle_semaphore: threading.Semaphore   = threading.Semaphore()
    stop_event: threading.Event             = threading.Event()
    workers: list[AbstractWorker]           = []

    @classmethod
    def start(cls) -> None:

        # Add genesis transaction to tangle
        tx: Transaction = {
            'tx_id': 'genesis',
            'approved_tx_ids': [],
            'state_dict': CNN(cls.dataset_manager.num_classes).to(cls.device).state_dict(),
            'creator_id': 'genesis',
            'timestamp': datetime.now().isoformat()
        }
        save_state_dict_to_disk(cls.id, tx['tx_id'], tx['state_dict'])
        attr = { 'creator_id': tx['creator_id'], 'timestamp': tx['timestamp'] }
        cls.tangle.add_node(tx['tx_id'], **attr)

        # Start honest trainers
        for i in range(cls.concurrent_honest_workers):
            w = cls.approach['trainer'](id=i+1)
            w.start()
            cls.workers.append(w)

        # If attack scenario is specified, start attacker
        if cls.attack_scenario:
            w = cls.attack_scenario['attacker'](id=1)
            w.start()
            cls.workers.append(w)

    @classmethod
    def stop(cls) -> None:
        cls.stop_event.set()
        print("\n[FLSimulation] Signaling all workers to stop.")
        for w in cls.workers:
            w.join()
        print("\n[FLSimulation] All workers have stopped.")

        # Start evaluator
        w = cls.approach['evaluator'](id=1)
        w.start()
