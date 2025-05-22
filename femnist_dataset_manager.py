from __future__ import annotations
import datasets
from torch.utils.data import DataLoader, Dataset
from statistics import mean
from typing import List, Tuple, Dict
import hashlib
from torchvision import transforms
from PIL import Image
import io
import os
import random

class FEMNISTDatasetManager():

    num_classes=62

    def __init__(self, writer_ids_per_client: int, batch_size: int):
        self.writer_ids_per_client = writer_ids_per_client
        self.batch_size = batch_size
        self.train_datasets = {}
        self.test_datasets = {}
        dataset_path = './femnist'
        if os.path.exists(dataset_path):
            print("[FemnistDatasetManager] Loading FEMNIST dataset from disk...")
            load_writer_datasets = lambda split: {
                writer_id: datasets.load_from_disk(os.path.join(dataset_path, split, writer_id))
                for writer_id in os.listdir(os.path.join(dataset_path, split))
                if os.path.isdir(os.path.join(dataset_path, split, writer_id))
            }
            self.train_datasets = load_writer_datasets('train')
            self.test_datasets = load_writer_datasets('test')
        else:
            print("[FemnistDatasetManager] Downloading FEMNIST dataset, partitioning by writer_id, splitting 80/20 and saving to disk...")
            dataset = datasets.load_dataset('flwrlabs/femnist', split='train')
            df = dataset.to_pandas() #type: ignore

            for writer_id, group in df.groupby('writer_id'): #type: ignore
                ds = datasets.Dataset.from_pandas(group)
                train_ds, test_ds = ds.train_test_split(test_size=0.2, seed=42).values()
                train_ds.save_to_disk(f'{dataset_path}/train/{writer_id}')
                test_ds.save_to_disk(f'{dataset_path}/test/{writer_id}')
                self.train_datasets[writer_id] = train_ds
                self.test_datasets[writer_id] = test_ds

        writer_ids = list(self.test_datasets.keys())
        self.chunked_writer_ids = [
            writer_ids[i : i + writer_ids_per_client]
            for i in range(0, len(writer_ids), writer_ids_per_client)
        ]

    def get_random_data_loaders(self) -> Tuple[str, DataLoader, DataLoader]:
        return self.initialize_data_loaders(random.choice(self.chunked_writer_ids))

    def initialize_data_loaders(self, writer_ids: List[str]) -> Tuple[str, DataLoader, DataLoader]:
        """
        Create train and test DataLoaders for a set of writer_ids and generate a unique client identifier.

        Args:
            writer_ids (List[str]): List of writer IDs whose data should be contained in the data loaders.

        Returns:
            Tuple[str, DataLoader, DataLoader]: A tuple containing:
                - A short SHA-256 hash string based on the writer_ids representing the client ID.
                - DataLoader for the combined training dataset.
                - DataLoader for the combined test dataset.
        """
        sorted_ids = sorted(writer_ids)
        id_string = ",".join(sorted_ids)
        client_id = hashlib.sha256(id_string.encode()).hexdigest()[:5]

        train_ds = datasets.concatenate_datasets([self.train_datasets[writer_id] for writer_id in writer_ids]) #type: ignore
        test_ds = datasets.concatenate_datasets([self.test_datasets[writer_id] for writer_id in writer_ids]) #type: ignore

        train_loader = DataLoader(self.FEMNISTDataset(train_ds), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.FEMNISTDataset(test_ds), batch_size=self.batch_size, shuffle=False)

        return client_id, train_loader, test_loader

    def get_global_test_loader(self):
        test_ds = datasets.concatenate_datasets(list(self.test_datasets.values())) #type: ignore
        test_loader = DataLoader(self.FEMNISTDataset(test_ds), batch_size=self.batch_size, shuffle=False)
        return test_loader

    def yield_client_test_loaders(self):
        for writer_ids in self.chunked_writer_ids:
            client_id, _, test_loader = self.initialize_data_loaders(writer_ids)
            yield client_id, test_loader

    def get_dataset_information(self) -> Dict[str, int | float]:
        """
        Get summary statistics for the train and test datasets.

        Returns:
            Dict[str, float]: Dictionary with keys:
                - 'unique_partitions': Number of unique dataset partitions (writer_ids).
                - 'train_total': Total number of training examples.
                - 'train_max': Maximum number of examples in a single training partition.
                - 'train_min': Minimum number of examples in a single training partition.
                - 'train_avg': Average number of examples per training partition.
                - 'test_total': Total number of test examples.
                - 'test_max': Maximum number of examples in a single test partition.
                - 'test_min': Minimum number of examples in a single test partition.
                - 'test_avg': Average number of examples per test partition.
        """
        train_lengths = [len(ds) for ds in self.train_datasets.values()]
        test_lengths = [len(ds) for ds in self.test_datasets.values()]
        return {
            'unique_partitions': len(self.train_datasets),
            'train_total': sum(train_lengths),
            'train_max': max(train_lengths),
            'train_min': min(train_lengths),
            'train_avg': mean(train_lengths),
            'test_total': sum(test_lengths),
            'test_max': max(test_lengths),
            'test_min': min(test_lengths),
            'test_avg': mean(test_lengths),
        }

    class FEMNISTDataset(Dataset):
        """
        A pytorch dataset for the FEMNIST data, where the image is grayscaled, resized,
        converted to tensors and normalized.

        Args:
            data (list): List of examples, where each example is a dict with 'image' and 'character' keys.

        Methods:
            __len__(): Return the number of examples in the dataset.
            __getitem__(idx): Return a tuple (processed_image, character_label) at the given index.
        """
        def __init__(self, data: Dataset):
            self.data = data
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        def __len__(self):
            return len(self.data) #type: ignore

        def __getitem__(self, idx):
            item = self.data[idx]
            image = Image.open(io.BytesIO(item['image']['bytes']))
            image = self.transform(image)
            character = item['character']
            return image, character
