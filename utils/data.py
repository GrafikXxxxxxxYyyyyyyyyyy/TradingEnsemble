import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader



class TradingDataset (Dataset):
    def __init__(self, root_dir, train_len=180, test_len=30):
        self.root_dir = root_dir
        self.train_len = train_len
        self.test_len = test_len

        self.file_list = self._get_file_list()

        return
    

    def _get_file_list(self):
        file_list = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.csv'):
                        file_list.append(os.path.join(class_path, file_name))
                        
        return file_list
    

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        csv_file_path = self.file_list[idx]
        data = pd.read_csv(csv_file_path)
        
        # Убираем колонку с датами
        data = data.drop(columns=['Date'])

        # Разделяем выборку на train/test
        train = data.iloc[-(self.train_len + self.test_len) : -self.test_len].to_numpy()
        test = data['Close'].iloc[-self.test_len:].to_numpy()[:, None]

        # Транспонируем и переводим в тензорный формат [channels, seq_len]
        train = torch.tensor(train.transpose(1, 0), dtype=torch.float32)
        test = torch.tensor(test.transpose(1, 0), dtype=torch.float32)

        return train, test
