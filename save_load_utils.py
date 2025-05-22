import networkx as nx
import os
import gzip
import torch
import pickle
import json
from typing import Generator
from project_types import StateDict

def save_state_dict_to_disk(simulation_id: str, tx_id: str, state_dict: StateDict) -> None:
    path = f'simulations/{simulation_id}/models/{tx_id}.pth.gz'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        torch.save(state_dict, f) #type: ignore

def load_state_dict_from_disk(simulation_id: str, tx_id: str) -> StateDict:
    path = f'simulations/{simulation_id}/models/{tx_id}.pth.gz'
    with gzip.open(path, 'rb') as f:
        return torch.load(f) #type: ignore

def save_tangle_to_disk(simulation_id: str, tangle: nx.DiGraph) -> None:
    path = f'simulations/{simulation_id}/tangles/{len(tangle)}.gpickle'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tangle, f)

def yield_tangles_from_disk(simulation_id: str) -> Generator[nx.DiGraph, None, None]:
    path = f'simulations/{simulation_id}/tangles'
    gpickle_files = [f for f in os.listdir(path) if f.endswith(".gpickle")]
    gpickle_files.sort(key=lambda f: int(f.replace(".gpickle", "")))
    for file_name in gpickle_files:
        with open(f'{path}/{file_name}', 'rb') as f:
            yield pickle.load(f)

def save_metrics_to_disk(simulation_id: str, metrics: dict) -> None:
    with open(os.path.join(f'simulations/{simulation_id}', 'metrics.json'), 'w') as f:
        f.write(json.dumps(metrics, indent=2))
        f.write("\n")
