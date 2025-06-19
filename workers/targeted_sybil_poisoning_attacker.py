import threading
import time
from uuid import uuid4
from datetime import datetime

from fl_simulation import FLSimulation
from workers import TangleFLTrainer

from project_types import Transaction

class TargetedSybilPoisoningAttacker(TangleFLTrainer):

    def run(self) -> None:
        print(f"[{threading.current_thread().name}] Started.")
        last_injection_at = FLSimulation.attack_scenario['injection_start'] - FLSimulation.attack_scenario['injection_interval'] # first injection at 0 TXs #type: ignore
        _, triggered_train_loader, test_loader = FLSimulation.dataset_manager.get_triggered_dataloaders() # has 5x the amount of writer_ids than a honest trainer
        while not FLSimulation.stop_event.is_set():
            local_tangle = self.get_local_copy_of_tangle()
            if len(local_tangle) >= last_injection_at + FLSimulation.attack_scenario['injection_interval']: #pyright: ignore

                tx_ids_selected, w_selected          = self.tip_selection(local_tangle, test_loader)
                w_avg                                = self.average_models(w_selected)
                w_trained                            = self.train_model(w_avg, triggered_train_loader)

                print(f"\n[{threading.current_thread().name}] Injecting {FLSimulation.attack_scenario['num_sybils_per_injection']} targeted poisoned model(s)...") #pyright: ignore
                for _ in range(FLSimulation.attack_scenario['num_sybils_per_injection']):
                    tx: Transaction = {
                        'tx_id': str(uuid4()),
                        'approved_tx_ids': tx_ids_selected,
                        'state_dict': w_trained,
                        'creator_id': 'attacker',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.add_tx_to_tangle(tx)
                print(f"\n[{threading.current_thread().name}] {FLSimulation.attack_scenario['num_sybils_per_injection']} targeted poisoned model(s) injected.") #pyright: ignore

                last_injection_at += FLSimulation.attack_scenario['injection_interval'] #pyright: ignore
            time.sleep(1)
