import os
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from torchvision.utils import make_grid



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



#######################################################################################################################
class Logger ():
    def __init__ (self, 
                  test_dataset, 
                  loss_function,
                  metric=None,
                  delimeter=100,
                  table_delimeter=10):
        
        self.step = 0
        self.table_step = 0
        self.delim = delimeter
        self.table_delim = table_delimeter
        
        self.dataset = test_dataset
        self.loss_function = loss_function
        self.metric = metric
        
        return
    
    
    
    def __call__ (self, model, train_loss):
        self.step += 1
        
        # Логгируем ошибку на батче тренировочной выборки
        wandb.log({"Loss on TRAIN": train_loss})
        
        # Каждые delim шагов логгируем результаты, полученные на тестовой выборке 
        if self.step % self.delim == 0:
            test_loss = 0
            total_len = 0
            metric_loss = 0
            for it, (x_batch_test, y_batch_test) in enumerate(self.dataset):
                output = model(x_batch_test)
                
                if self.metric is not None:
                    metric_loss += self.metric(output, y_batch_test).cpu().item()*len(y_batch_test)
                
                test_loss += self.loss_function(output, y_batch_test).cpu().item()*len(y_batch_test)
                total_len += len(y_batch_test)

                # Временное решение останавливать просмотр тестовой выборки на том же кол-ве 
                # батчей, сколько посмотрели в тренировочной
                if it > self.delim:
                    break
                
            
            # Если задана метрика, то логгируем
            if self.metric is not None:
                wandb.log({"Metric MAE": metric_loss / total_len})
            
            # Логгируем ошибку на тестовой выборке
            wandb.log({"Loss on TEST": test_loss / total_len})

            # Логгируем изображение
            fig, axs = plt.subplots(2, 4, figsize=(40, 10))
            for i in range(8):
                row = i // 4
                col = i % 4

                real = torch.hstack((x_batch_test[i, 3, :], y_batch_test[i, 0, :]))
                predict = torch.hstack((x_batch_test[i, 3, :], output[i, 0, :]))

                axs[row, col].plot(predict, "red")
                axs[row, col].plot(real, "black")
                axs[row, col].grid()
            
            image_path = 'plot_image.png'
            plt.savefig(image_path)
            wandb.log({f"Step {self.step}": wandb.Image(image_path)})
            os.remove(image_path)
            plt.close()
            
            # Сохраняем структуру самой модели
            torch.onnx.export(model, x_batch_test, "model.onnx")
            wandb.save("model.onnx")
        
        return
    


    # def log_table (self, model):
    #     X, y = next(iter(self.dataset))

    #     # Если достигли предела таблицы, то логгируем
    #     if self.table_step % self.table_delim == 0:
    #         wandb.log({f"Intermediate results {self.table_step - self.table_delim}-{self.table_step}": self.intermediate_results})

    #     pass
    
    

    def stop_logging (self):
        """
        Данная функция просто завершает процесс логирования и выводит всю информацию,
        которая копилась в процессе (таблицы, onnx и пр.)
        """
        # В нашем случае просто логгируем табличку с промежуточными картинками
        wandb.log({"Intermediate results": self.intermediate_results})
        wandb.finish()

        return 
#######################################################################################################################



#######################################################################################################################
# Функция обучения на отдельном батче
def train_batch (x_batch, y_batch, model, optimizer, loss_function):
    # Переводим модель в режим обучения
    model.train()
    
    # Обнуляем остатки градиентов с прошлых шагов
    optimizer.zero_grad()
    
    # Строим прогноз по батчу
    output = model(x_batch)
    
    # Ошибка между output и y_batch
    loss = loss_function(output, y_batch)
    loss.backward()
    
    # Делаем шаг оптимизатора
    optimizer.step()
    
    return loss.cpu().item()
#######################################################################################################################



#######################################################################################################################
# Функция обучения на одной эпохе
def train_epoch (train_loader, model, optimizer, loss_function, logger=None):
    epoch_loss = 0
    total_len = 0
    
    # цикл по генератору train (по батчам)
    for it, (x_batch, y_batch) in enumerate(train_loader):
        # Вызов train_batch
        batch_loss = train_batch(x_batch, y_batch, model, optimizer, loss_function)
    
        # Логгируем модель если указан логгер
        if logger is not None:
            with torch.no_grad():
                logger(model, batch_loss)
        
        epoch_loss += batch_loss*len(x_batch)
        total_len += len(x_batch)

    return epoch_loss / total_len
#######################################################################################################################



#######################################################################################################################
# Функция для обучения модели
def trainer (train_loader, model, optimizer, loss_function, epochs=5, lr=1e-3, logger=None):
    # Загрузим параметры модели в оптимизатор
    optim = optimizer(model.parameters(), lr)
    
    tqdm_iterations = tqdm(range(epochs), desc='Epoch')
    tqdm_iterations.set_postfix({'Current train loss': np.nan})

    for epoch in tqdm_iterations:
        current_loss = train_epoch(train_loader, model, optim, loss_function, logger)
        
        tqdm_iterations.set_postfix({'Current train loss': current_loss})

    # Выводим всякую промежуточную информацию из логов
    if logger is not None:
        logger.stop_logging()        

    return
#######################################################################################################################