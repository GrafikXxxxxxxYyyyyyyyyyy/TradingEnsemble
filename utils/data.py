import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader



class TradingDataset (Dataset):
    def __init__(self, root_dir, train_len=256, test_len=32):
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



# Отрисовывает результаты предсказания модели
def show_quality (model, loader, num=4):
    train, test = next(iter(loader))
    
    train = train[:num]
    test = test[:num]
    
    pred = model(train).detach()

    fig, axs = plt.subplots(int(num**(1/2)), int(num**(1/2)), figsize=(20, 10))
    
    for i in range(num):
        row = i // int(num**(1/2))
        col = i % int(num**(1/2))

        real = torch.hstack((train[i, 3, :], test[i, 0, :]))
        predict = torch.hstack((train[i, 3, :], pred[i, 0, :]))

        axs[row, col].plot(predict, "red")
        axs[row, col].plot(real, "black")
        axs[row, col].grid()
    
    image_path = 'plot_image.png'
    plt.savefig(image_path)
    plt.show()

    os.remove(image_path)
    
    return




# Функция для отрисовки результатов
def show_data (loader):
    train, test = next(iter(loader))

    plt.figure(figsize=(20, 10))
    # Цены закрытия [3]
    plt.plot(torch.hstack((train[0, 3, :], test[0, 0, :])), "red")
    plt.plot(train[0, 3, :], "black")
    # Цены открытия/макс/мин/средние
    plt.plot(train[0, 0, :], label="Open")
    plt.plot(train[0, 1, :], label="High") 
    plt.plot(train[0, 2, :], label="Low") 
    plt.plot(train[0, 5, :], label="EMA 9") 
    plt.plot(train[0, 6, :], label="EMA 20")
    plt.plot(train[0, 7, :], label="EMA 50")
    plt.plot(train[0, 8, :], label="EMA 200")
    plt.legend()
    plt.title("Prices")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 4, :], label="Volume")
    plt.legend()
    plt.title("Volume normalized")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 9, :], label="Volatility 9")
    plt.plot(train[0, 10, :], label="Volatility 20")
    plt.plot(train[0, 11, :], label="Volatility 50")
    plt.plot(train[0, 12, :], label="Volatility 200")
    plt.legend()
    plt.title("Volatility")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 13, :], label="RSI 5")
    plt.plot(train[0, 14, :], label="RSI 9")
    plt.plot(train[0, 15, :], label="RSI 14")
    plt.legend()
    plt.title("RSI")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 16, :], label="MACD line")
    plt.plot(train[0, 17, :], label="MACD signal")
    plt.legend()
    plt.title("MACD")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 18, :], label="Stochastic %D 5")
    plt.plot(train[0, 19, :], label="Stochastic %D 9")
    plt.plot(train[0, 20, :], label="Stochastic %D 14")
    plt.legend()
    plt.title("Stochastic")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 21, :], label="RVI 5")
    plt.plot(train[0, 22, :], label="RVI 9")
    plt.plot(train[0, 23, :], label="RVI 14")
    plt.legend()
    plt.title("Relative Vigor Index (RVI)")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 24, :], label="Money Flow Index")
    plt.legend()
    plt.title("Money Flow Index")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(train[0, 25, :], label="Awesome Oscillator")
    plt.legend()
    plt.title("Awesome Oscillator")
    plt.grid()
    plt.show()

    return
