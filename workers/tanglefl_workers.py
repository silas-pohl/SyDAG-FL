from abc import ABC, abstractmethod
import threading
from uuid import uuid4
from datetime import datetime
import numpy as np
import random
import math
import networkx as nx
from decimal import Decimal
from collections import Counter
from queue import PriorityQueue
from torch.utils.data import DataLoader

from fl_simulation import FLSimulation
from workers.abstract_worker import AbstractWorker
from project_types import Transaction, StateDict
from save_load_utils import yield_tangles_from_disk, load_state_dict_from_disk, save_metrics_to_disk

class TangleFLBase(AbstractWorker, ABC):

    @abstractmethod
    def run(self) -> None:
        pass

    def biased_random_walk_until_tip(self, tangle: nx.DiGraph, tx_id: str, path: list[str] = []) -> list[str]:
        path.append(tx_id)
        direct_approvals = list(tangle.predecessors(tx_id))
        if not direct_approvals: return path
        else:
            weights = []
            for direct_approval in direct_approvals:
                base_weight = 1+len(list(nx.ancestors(tangle, direct_approval)))
                weights.append(math.pow(base_weight, FLSimulation.approach['biased_random_walk_alpha']))
            selected_tx = random.choices(direct_approvals, weights, k=1)[0]
            return self.biased_random_walk_until_tip(tangle, selected_tx, path)

    def determine_consensus(self, tangle: nx.DiGraph) -> StateDict:
        confidence = Counter()
        for i in range(FLSimulation.approach['sample_size_for_consensus']):
            path = self.biased_random_walk_until_tip(tangle, 'genesis', [])
            for tx_id in path:
                confidence[tx_id] += (Decimal(1)/Decimal(FLSimulation.approach['sample_size_for_consensus'])) #type: ignore

        prio_queue = PriorityQueue()
        for tx_id, confidence in confidence.items():
            prio = confidence*len(nx.descendants(tangle, tx_id)) #confidence * rating
            prio_queue.put((-prio, tx_id))

        top_n_state_dicts = []
        num_to_retrieve = min(FLSimulation.approach['consensus_based_on_top_n'], prio_queue.qsize())
        for _ in range(num_to_retrieve):
            top_n_state_dicts.append(load_state_dict_from_disk(FLSimulation.id, prio_queue.get()[1]))

        return self.average_models(top_n_state_dicts)

    def tip_selection(self, tangle: nx.DiGraph, test_loader: DataLoader) -> tuple[list[str], list[StateDict]]:
        sampled_tips = set()
        for _ in range(FLSimulation.approach['sample_size_for_tip_selection']):
            sampled_tips.add(self.biased_random_walk_until_tip(tangle, 'genesis', [])[-1])

        sampled_tips = list(sampled_tips)
        selected_tips = []
        if len(sampled_tips) > FLSimulation.approach['num_tips']:
            prio_queue = PriorityQueue()
            for tip in sampled_tips:
                state_dict = load_state_dict_from_disk(FLSimulation.id, tip)
                prio = self.evaluate_model(state_dict, test_loader)['accuracy']
                prio_queue.put((-prio, tip)) #type: ignore

            for _ in range(FLSimulation.approach['num_tips']):
                selected_tips.append(prio_queue.get()[1])
        else:
            selected_tips = sampled_tips

        selected_state_dicts = [load_state_dict_from_disk(FLSimulation.id, tip) for tip in selected_tips]

        return selected_tips, selected_state_dicts

class TangleFLTrainer(TangleFLBase):

    def run(self) -> None:
        print(f"[{threading.current_thread().name}] Started.")
        while not FLSimulation.stop_event.is_set():
            client_id, train_loader, test_loader = FLSimulation.dataset_manager.get_random_data_loaders()
            local_tangle                         = self.get_local_copy_of_tangle()                           # From AbstractWorker
            w_consensus                          = self.determine_consensus(local_tangle)
            tx_ids_selected, w_selected          = self.tip_selection(local_tangle, test_loader)
            w_avg                                = self.average_models(w_selected)                           # From AbstractWorker
            w_trained                            = self.train_model(w_avg, train_loader)                     # From AbstractWorker
            accuracy_trained                     = self.evaluate_model(w_trained, test_loader)['accuracy']   # From AbstractWorker
            accuracy_consensus                   = self.evaluate_model(w_consensus, test_loader)['accuracy'] # FRom AbstractWorker
            if accuracy_trained > accuracy_consensus: #type: ignore
                tx: Transaction = {
                    'tx_id': str(uuid4()),
                    'approved_tx_ids': tx_ids_selected,
                    'state_dict': w_trained,
                    'creator_id': client_id,
                    'timestamp': datetime.now().isoformat()
                }
                self.add_tx_to_tangle(tx)

class TangleFLEvaluator(TangleFLBase):

    def run(self) -> None:
        print(f"[{threading.current_thread().name}] Started.")
        metrics = {}
        for tangle in yield_tangles_from_disk(FLSimulation.id):
            w_consensus         = self.determine_consensus(tangle)

            print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: Evaluating performance metrics (global test dataset)...")
            test_loader         = FLSimulation.dataset_manager.get_global_test_loader()
            performance_metrics = {k:v for k,v in self.evaluate_model(w_consensus, test_loader)['weighted avg'].items() if k != 'support'} #type: ignore
            print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: {performance_metrics}")

            print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: Evaluating fairness metrics (client's test datasets)...")
            fairness_metrics = {}
            f1_scores = []
            for client_id, test_loader in FLSimulation.dataset_manager.yield_client_test_loaders():
                f1_scores.append(self.evaluate_model(w_consensus, test_loader)['weighted avg']['f1-score']) #type: ignore
            f1_scores = np.array(f1_scores)
            fairness_metrics['f1_min'] = float(f1_scores.min())
            fairness_metrics['f1_std'] = float(f1_scores.std())
            fairness_metrics['f1_jfi'] = float((np.sum(f1_scores) ** 2) / (len(f1_scores) * np.sum(f1_scores ** 2)))
            print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: {fairness_metrics}")

            metrics[len(tangle)] = {'performance_metrics': performance_metrics,'fairness_metrics': fairness_metrics}
            save_metrics_to_disk(FLSimulation.id, metrics)
