from abc import ABC, abstractmethod
import threading
import networkx as nx
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from fl_simulation import FLSimulation
from cnn import CNN
from project_types import Transaction, StateDict
from save_load_utils import save_state_dict_to_disk, save_tangle_to_disk

class AbstractWorker(threading.Thread, ABC):

    def __init__(self, id: int):
        super().__init__(name=f'{self.__class__.__name__}-{id}')

    @abstractmethod
    def run(self) -> None:
        pass

    def get_local_copy_of_tangle(self) -> nx.DiGraph:
        with FLSimulation.tangle_semaphore:
            return copy.deepcopy(FLSimulation.tangle)

    def add_tx_to_tangle(self, tx: Transaction) -> None:
        with FLSimulation.tangle_semaphore:
            save_state_dict_to_disk(FLSimulation.id, tx['tx_id'], tx['state_dict'])
            attr = { 'creator_id': tx['creator_id'], 'timestamp': tx['timestamp'] }
            FLSimulation.tangle.add_node(tx['tx_id'], **attr)
            for approved_tx_id in tx['approved_tx_ids']:
                FLSimulation.tangle.add_edge(tx['tx_id'], approved_tx_id)

            # Check evaluation_interval and save tangle to disk for later evaluation
            print(f"\rTangle size: {len(FLSimulation.tangle)}", end="", flush=True)
            if len(FLSimulation.tangle) % FLSimulation.evaluation_interval == 0:
                save_tangle_to_disk(FLSimulation.id, FLSimulation.tangle)

            if len(FLSimulation.tangle) == FLSimulation.stop_threshold:
                FLSimulation.stop_event.set()

    def average_models(self, state_dicts: list[StateDict]) -> StateDict:
        avg_weights = {}
        for key in state_dicts[0]:
            avg_weights[key] = torch.stack([
                state_dict[key].detach().to(FLSimulation.device).float()
                for state_dict in state_dicts
            ], dim=0).mean(dim=0)
        return avg_weights

    def train_model(self, state_dict: StateDict, train_loader: DataLoader) -> StateDict:
        model = CNN(FLSimulation.dataset_manager.num_classes).to(FLSimulation.device)
        model.load_state_dict(state_dict)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=FLSimulation.approach['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(FLSimulation.approach['epochs']):
            for x, y in train_loader:
                x, y = x.to(FLSimulation.device), y.to(FLSimulation.device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def evaluate_model(self, state_dict: StateDict, test_loader: DataLoader) -> dict[str, float | dict[str, float]]:
        model = CNN(FLSimulation.dataset_manager.num_classes).to(FLSimulation.device)
        model.load_state_dict(state_dict)
        model.eval()
        all_target, all_preds = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(FLSimulation.device), y.to(FLSimulation.device)
                outputs = model(x)
                _, preds = torch.max(outputs.data, 1)
                all_target.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        return classification_report(all_target, all_preds, digits=4, output_dict=True, zero_division=0) # type: ignore
