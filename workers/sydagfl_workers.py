from abc import ABC, abstractmethod
import threading
from uuid import uuid4
from datetime import datetime
import torch
import numpy as np
import random
import math
import networkx as nx
from decimal import Decimal
from collections import Counter, defaultdict
from queue import PriorityQueue
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy

from fl_simulation import FLSimulation
from workers.abstract_worker import AbstractWorker
from project_types import Transaction, StateDict
from save_load_utils import yield_tangles_from_disk, load_state_dict_from_disk, save_metrics_to_disk

class SyDAGFLBase(AbstractWorker, ABC):

    @abstractmethod
    def run(self) -> None:
        pass

    def flatten_state_dict(self, state_dict: StateDict) -> torch.Tensor:
        return torch.cat([v.flatten() for v in state_dict.values()])

    def get_or_compute_similarity(self, tx1: str, tx2: str) -> float:

        key = ''.join(sorted([tx1, tx2]))
        if key in FLSimulation.tangle.graph['similarity_cache']:
            cosine_sim = FLSimulation.tangle.graph['similarity_cache'][key]
        else:
            vec1 = self.flatten_state_dict(load_state_dict_from_disk(FLSimulation.id, tx1)).to(FLSimulation.device)
            vec2 = self.flatten_state_dict(load_state_dict_from_disk(FLSimulation.id, tx2)).to(FLSimulation.device)
            cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
            FLSimulation.tangle.graph['similarity_cache'][''.join(sorted([tx1, tx2]))] = cosine_sim
        return cosine_sim

    def assign_group_to_transaction(self, tx: str, tangle: nx.DiGraph, similarity_threshold: float) -> nx.DiGraph:
        group_assigned = False
        for existing_tx in list(tangle.nodes)[-30:]: # Compare with 30 latest nodes
            if existing_tx == tx: continue  # Skip the current node when comparing

            similarity = self.get_or_compute_similarity(tx, existing_tx)

            if similarity >= similarity_threshold:
                if 'similarity_group' in tangle.nodes[existing_tx]:
                    tangle.nodes[tx]['similarity_group'] = tangle.nodes[existing_tx]['similarity_group']  # Assign the same group
                    group_assigned = True
                    print("FOUND TWO SIMILAR UPDATES")
                    #print(f"\n[{threading.current_thread().name}] SAME SIMILARITY GROUP FOUND")
                    break

        if not group_assigned:
            new_group = len(tangle.nodes)  # If no similar group is found, assign a new group
            tangle.nodes[tx]['similarity_group'] = new_group #

        return tangle

    def assign_similarity_groups(self, tangle: nx.DiGraph, similarity_threshold: float) -> nx.DiGraph:

        for tx in tangle.nodes:
            if 'similarity_group' not in tangle.nodes[tx]:  # Check if the node doesn't already have a group
                #print(f"[{threading.current_thread().name}] Assigning similarity group for {tx}...")
                tangle = self.assign_group_to_transaction(tx, tangle, similarity_threshold)

        return tangle

    # Overwrite get_local_copy_of_tangle of abstract_worker to assign_similarity_groups to global tangle whenever the tangle is loaded
    def get_local_copy_of_tangle(self) -> nx.DiGraph:
        with FLSimulation.tangle_semaphore:
            FLSimulation.tangle.graph.setdefault('similarity_cache', {})
            FLSimulation.tangle = self.assign_similarity_groups(FLSimulation.tangle, 0.999999)
            return copy.deepcopy(FLSimulation.tangle)

    def sybil_aware_biased_random_walk_until_tip(self, tangle: nx.DiGraph, tx_id: str, path: list[str] = []) -> list[str]:
        path.append(tx_id)
        potential_next_steps = list(tangle.predecessors(tx_id))
        if not potential_next_steps: return path
        else:
            weights = []
            for i, potential_next_step in enumerate(potential_next_steps):
                approvals = nx.ancestors(tangle, potential_next_step)

                # Calculate the base weight as the number of unique similarity groups that approved this transaction
                base_weight = 1+len({tangle.nodes[approval]['similarity_group'] for approval in approvals})

                # Adjust base weight if other potential next steps are in the same similarity group
                same_group_count = sum(
                    1 for other_next_step in potential_next_steps
                    if tangle.nodes[other_next_step]['similarity_group'] == tangle.nodes[potential_next_step]['similarity_group']
                )
                base_weight *= (1 / same_group_count)

                weights.append(math.pow(base_weight, FLSimulation.approach['biased_random_walk_alpha']))
            selected_tx = random.choices(potential_next_steps, weights, k=1)[0]
            return self.sybil_aware_biased_random_walk_until_tip(tangle, selected_tx, path)

    def sybil_aware_consensus_selection(self, tangle: nx.DiGraph) -> StateDict:
        confidence = Counter()
        for i in range(FLSimulation.approach['sample_size_for_consensus']):
            path = self.sybil_aware_biased_random_walk_until_tip(tangle, 'genesis', [])
            for tx_id in path:
                confidence[tx_id] += (Decimal(1)/Decimal(FLSimulation.approach['sample_size_for_consensus'])) #type: ignore

        prio_queue = PriorityQueue()
        for tx_id, confidence in confidence.items():
            prio = confidence*len(nx.descendants(tangle, tx_id)) #confidence * rating
            prio_queue.put((-prio, tx_id))

        # Get top TXs from prio queue grouped by similarity_group until prio_queue is empty
        # or {consensus_based_on_top_n} different similarity groups are reached
        consensus_basis_dict: defaultdict[int, list[str]] = defaultdict(list)
        while (not prio_queue.empty()) and (len(consensus_basis_dict) < FLSimulation.approach['consensus_based_on_top_n']):
            tx_id = prio_queue.get()[1]
            consensus_basis_dict[tangle.nodes[tx_id]['similarity_group']].append(tx_id)

        # Average all tx_ids in the same similarity_group and add the averaged state_dict to the final array
        consensus_basis_state_dicts = []
        for _, tx_ids in consensus_basis_dict.items():
            state_dicts = [load_state_dict_from_disk(FLSimulation.id, tx_id) for tx_id in tx_ids]
            if len(state_dicts) > 1:
                state_dict = self.average_models(state_dicts)
            else:
                state_dict = state_dicts[0]
            consensus_basis_state_dicts.append(state_dict)

        return self.average_models(consensus_basis_state_dicts)

    def sybil_aware_tip_selection(self, tangle: nx.DiGraph, test_loader: DataLoader) -> tuple[list[str], list[StateDict]]:

        # Conduct Sybil-aware biased random walks and group the tips by similarity group
        sampled_tips_grouped_by_similarity_group: defaultdict[int, set[str]] = defaultdict(set)
        for _ in range(FLSimulation.approach['sample_size_for_tip_selection']):
            tx_id = self.sybil_aware_biased_random_walk_until_tip(tangle, 'genesis', [])[-1]
            sampled_tips_grouped_by_similarity_group[tangle.nodes[tx_id]['similarity_group']].add(tx_id)

        # Average all sampled tips of the same similarity_group
        sampled_tips_similarity_groups_averaged: list[tuple[list[str], StateDict]] = []
        for similarity_group, tx_ids in sampled_tips_grouped_by_similarity_group.items():
            state_dicts = [load_state_dict_from_disk(FLSimulation.id, tx_id) for tx_id in tx_ids]
            if len(state_dicts) > 1:
                state_dict = self.average_models(state_dicts)
            else:
                state_dict = state_dicts[0]
            sampled_tips_similarity_groups_averaged.append((list(tx_ids), state_dict))

        # If more tips sampled than {num_tips}, get the top {num_tips} tips (or averaged tips)
        selected_tips_and_state_dicts = []
        if len(sampled_tips_similarity_groups_averaged) > FLSimulation.approach['num_tips']:
            prio_queue = PriorityQueue()
            for tx_ids, state_dict in sampled_tips_similarity_groups_averaged:
                prio = self.evaluate_model(state_dict, test_loader)['accuracy']
                prio_queue.put((-prio, (tx_ids, state_dict))) #type: ignore
            for _ in range(FLSimulation.approach['num_tips']):
                selected_tips_and_state_dicts.append(prio_queue.get()[1])
        else:
            selected_tips_and_state_dicts = sampled_tips_similarity_groups_averaged

        # Flatten list of lists of selected_tips
        selected_tips = [tx_id for tx_ids, _ in selected_tips_and_state_dicts for tx_id in tx_ids]

        # Get state_dicts
        selected_state_dicts = [state_dict for _, state_dict in selected_tips_and_state_dicts ]

        return selected_tips, selected_state_dicts

class SyDAGFLTrainer(SyDAGFLBase):

        def run(self) -> None:
            print(f"[{threading.current_thread().name}] Started.")
            while not FLSimulation.stop_event.is_set():
                client_id, train_loader, test_loader = FLSimulation.dataset_manager.get_random_data_loaders()
                local_tangle                         = self.get_local_copy_of_tangle()                           # OVERWRITTEN
                w_consensus                          = self.sybil_aware_consensus_selection(local_tangle)
                tx_ids_selected, w_selected          = self.sybil_aware_tip_selection(local_tangle, test_loader)
                w_avg                                = self.average_models(w_selected)                           # From AbstractWorker
                w_trained                            = self.train_model(w_avg, train_loader)                     # From AbstractWorker
                accuracy_trained                     = self.evaluate_model(w_trained, test_loader)['accuracy']   # From AbstractWorker
                accuracy_consensus                   = self.evaluate_model(w_consensus, test_loader)['accuracy'] # From AbstractWorker
                if accuracy_trained > accuracy_consensus: #type: ignore
                    tx: Transaction = {
                        'tx_id': str(uuid4()),
                        'approved_tx_ids': tx_ids_selected,
                        'state_dict': w_trained,
                        'creator_id': client_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.add_tx_to_tangle(tx)
                else:
                    FLSimulation.no_improvement_counter += 1
                    print(f"\rTangle size: {len(FLSimulation.tangle)}; No improvements: {FLSimulation.no_improvement_counter}", end="", flush=True)
                    if FLSimulation.no_improvement_counter >= 1000:
                        FLSimulation.stop_event.set()

class SyDAGFLEvaluator(SyDAGFLBase):

    def run(self) -> None:
        print(f"[{threading.current_thread().name}] Started.")
        metrics = {}
        for tangle in yield_tangles_from_disk(FLSimulation.id):
            tangle = self.assign_similarity_groups(tangle, 0.999999)

            w_consensus = self.sybil_aware_consensus_selection(tangle)

            if True:
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: Evaluating performance metrics (global test dataset)...")
                test_loader         = FLSimulation.dataset_manager.get_global_test_loader()
                performance_metrics = {k:v for k,v in self.evaluate_model(w_consensus, test_loader)['weighted avg'].items() if k != 'support'} #type: ignore
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: {performance_metrics}")
            else:
                performance_metrics = {}

            if True:
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: Evaluating fairness metrics (client's test datasets)...")
                fairness_metrics = {}
                f1_scores = []
                for test_loader in FLSimulation.dataset_manager.yield_client_test_loaders(0.1): # Test with 10% of all client test_loaders
                    f1_scores.append(self.evaluate_model(w_consensus, test_loader)['weighted avg']['f1-score']) #type: ignore
                f1_scores = np.array(f1_scores)
                fairness_metrics['f1_min'] = float(f1_scores.min())
                fairness_metrics['f1_std'] = float(f1_scores.std())
                fairness_metrics['f1_jfi'] = float((np.sum(f1_scores) ** 2) / (len(f1_scores) * np.sum(f1_scores ** 2)))
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: {fairness_metrics}")
            else:
                fairness_metrics = {}

            if FLSimulation.attack_scenario.get('asr'): #type: ignore
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: Evaluating ASR (global triggered test dataset)...")
                triggered_test_loader = FLSimulation.dataset_manager.get_triggered_global_test_loader()
                asr = self.evaluate_model(w_consensus, triggered_test_loader)['accuracy']
                print(f"[{threading.current_thread().name}] {len(tangle):>4} TX: ASR = {asr}")
                metrics[len(tangle)] = {'performance_metrics': performance_metrics,'fairness_metrics': fairness_metrics, 'asr': asr}
            else:
                metrics[len(tangle)] = {'performance_metrics': performance_metrics,'fairness_metrics': fairness_metrics}
            save_metrics_to_disk(FLSimulation.id, metrics, FLSimulation.approach, FLSimulation.attack_scenario) #type: ignore
