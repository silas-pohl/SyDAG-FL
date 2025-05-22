import time
from datetime import datetime

import workers
from fl_simulation import FLSimulation
from femnist_dataset_manager import FEMNISTDatasetManager

untargeted_sybil_poisoning_scenario = {
    'attacker': workers.UntargetedSybilPoisoningAttacker,
}

targeted_sybil_poisoning_scenario = {
    'attacker': workers.TargetedSybilPoisoningAttacker,
}

sydag_fl = {
    'trainer': workers.SyDAGFLTrainer,
    'evaluator': workers.SyDAGFLEvaluator,
    'consensus_based_on_top_n': 10,
    'sample_size_for_consensus': 35,
    'sample_size_for_tip_selection': 35,
    'num_tips': 3,
    'biased_random_walk_alpha': 1.0,
    'epochs': 1,
    'learning_rate': 0.006,
}

tangle_fl = {
    'trainer': workers.TangleFLTrainer,
    'evaluator': workers.TangleFLEvaluator,
    'consensus_based_on_top_n': 10,
    'sample_size_for_consensus': 35,
    'sample_size_for_tip_selection': 35,
    'num_tips': 3,
    'biased_random_walk_alpha': 1.0,
    'epochs': 1,
    'learning_rate': 0.006,
}

FLSimulation.id = datetime.now().isoformat(timespec='seconds')
FLSimulation.approach = tangle_fl
FLSimulation.attack_scenario = None
FLSimulation.concurrent_honest_workers = 5
FLSimulation.evaluation_interval = 20
FLSimulation.stop_threshold = 1000
FLSimulation.dataset_manager = FEMNISTDatasetManager(1, 32)

FLSimulation.start()
try:
    while not FLSimulation.stop_event.is_set(): time.sleep(0.1)
    FLSimulation.stop()
except KeyboardInterrupt:
    FLSimulation.stop()
