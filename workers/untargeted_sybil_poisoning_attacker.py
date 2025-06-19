import random
import torch
from uuid import uuid4
from datetime import datetime
import threading
import time
import networkx as nx

from workers.abstract_worker import AbstractWorker
from fl_simulation import FLSimulation
from cnn import CNN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from project_types import Transaction

class UntargetedSybilPoisoningAttacker(AbstractWorker):

    def run(self) -> None:
        print(f"[{threading.current_thread().name}] Started.")
        last_injection_at = FLSimulation.attack_scenario['injection_start'] - FLSimulation.attack_scenario['injection_interval'] #type: ignore
        while not FLSimulation.stop_event.is_set():
            local_tangle = self.get_local_copy_of_tangle()
            if len(local_tangle) >= last_injection_at + FLSimulation.attack_scenario['injection_interval']: #pyright: ignore

                w_random_noise = CNN(FLSimulation.dataset_manager.num_classes).state_dict()
                for key in w_random_noise:
                    w_random_noise[key] = torch.randn_like(w_random_noise[key])

                all_tips = [tx for tx in local_tangle if local_tangle.in_degree(tx) == 0]
                tx_ids_selected = random.choices(all_tips, k=FLSimulation.approach['num_tips'])

                print(f"\n[{threading.current_thread().name}] Injecting {FLSimulation.attack_scenario['num_sybils_per_injection']} sybil untargeted poisoned models.") #pyright: ignore
                for _ in range(FLSimulation.attack_scenario['num_sybils_per_injection']):
                    tx: Transaction = {
                        'tx_id': str(uuid4()),
                        'approved_tx_ids': tx_ids_selected,
                        'state_dict': w_random_noise,
                        'creator_id': 'attacker',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.add_tx_to_tangle(tx)
                print(f"\n[{threading.current_thread().name}] {FLSimulation.attack_scenario['num_sybils_per_injection']} targeted poisoned model(s) injected.") #pyright: ignore

                last_injection_at += FLSimulation.attack_scenario['injection_interval'] #pyright: ignore
            time.sleep(1)
