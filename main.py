import time
import sys

import workers
from fl_simulation import FLSimulation
from femnist_dataset_manager import FEMNISTDatasetManager

FLSimulation.id = sys.argv[1] + '-' + sys.argv[2]

match sys.argv[1]:
    case 'tangle-fl':
        FLSimulation.approach = {
            'trainer': workers.TangleFLTrainer,
            'evaluator': workers.TangleFLEvaluator,
            'consensus_based_on_top_n': 10,
            'sample_size_for_consensus': 100,
            'sample_size_for_tip_selection': 100,
            'num_tips': 3,
            'biased_random_walk_alpha': 1.0,
            'epochs': 1,
            'learning_rate': 0.006,
        }
    case 'sydag-fl':
        FLSimulation.approach = {
            'trainer': workers.SyDAGFLTrainer,
            'evaluator': workers.SyDAGFLEvaluator,
            'consensus_based_on_top_n': 10,
            'sample_size_for_consensus': 100,
            'sample_size_for_tip_selection': 100,
            'num_tips': 3,
            'biased_random_walk_alpha': 1.0,
            'epochs': 1,
            'learning_rate': 0.006,
        }
    case _:
        raise Exception('Please specify approach: tangle-fl or sydag-fl')

match sys.argv[2]:
    case 'non-adversarial':
        FLSimulation.attack_scenario = {}
    case 'untargeted1':
        FLSimulation.attack_scenario = {
            'attacker': workers.UntargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 1
        }
    case 'untargeted5':
        FLSimulation.attack_scenario = {
            'attacker': workers.UntargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 5
        }
    case 'untargeted25':
        FLSimulation.attack_scenario = {
            'attacker': workers.UntargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 25
        }
    case 'untargeted50':
        FLSimulation.attack_scenario = {
            'attacker': workers.UntargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 50
        }
    case 'targeted1':
        FLSimulation.attack_scenario = {
            'attacker': workers.TargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 1,
            'asr': True
        }
    case 'targeted5':
        FLSimulation.attack_scenario = {
            'attacker': workers.TargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 5,
            'asr': True
        }
    case 'targeted25':
        FLSimulation.attack_scenario = {
            'attacker': workers.TargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 25,
            'asr': True
        }
    case 'targeted50':
        FLSimulation.attack_scenario = {
            'attacker': workers.TargetedSybilPoisoningAttacker,
            'injection_start': 250,
            'injection_interval': 250,
            'num_sybils_per_injection': 50,
            'asr': True
        }
    case _:
        raise Exception('Please specify attack scenario: non-adversarial, untargeted1/25 or targeted1/25')

FLSimulation.concurrent_honest_workers = 5
FLSimulation.evaluation_interval = 50
FLSimulation.stop_threshold = 1000
FLSimulation.dataset_manager = FEMNISTDatasetManager(1, 32)

FLSimulation.start()
try:
    while not FLSimulation.stop_event.is_set(): time.sleep(0.1)
    FLSimulation.stop()
except KeyboardInterrupt:
    FLSimulation.stop()
