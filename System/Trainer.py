import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from Model.MyDataset import MyDataset
from System.Utilis import loadJsonFile
from Model.NILMFormer.mossformer import MossFormer


class Trainer(threading.Thread):
    def __init__(self, config, historyDataPath):
        super().__init__()
        self.config = config
        self.historyDataPath = historyDataPath
        self.trainLoader = None
        self.testLoader = None

        currentTime = datetime.now()
        timeString = currentTime.strftime("%Y-%m-%d_%H-%M-%S")

        self.checkpoint = os.path.join(config['train']['checkpoint'], timeString)
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            self.device = torch.device("cuda:0")
        else:
            torch.manual_seed(42)
            self.device = torch.device("cpu")

        self.model = MossFormer(self.config['initialize'])
        self.model.to(self.device)
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def run(self) -> None:
        self.train()

    def train(self):
        TrainLoader, TestLoader = self.createDataloader(self.readHistoryData())
        lr = self.config['lr']
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        MinMAE = float("inf")

        for epoch in range(1, self.config['epoch'] + 1):
            MSELossTrain = 0
            self.model.train()
            for idx, (main, target) in enumerate(TrainLoader):
                main = main.type(torch.FloatTensor).to(device=self.device)
                target = target.type(torch.FloatTensor).to(device=self.device)
                target = target.transpose(0, 1)
                ReconLoss = torch.tensor(0.0).to(device=self.device)

                optimizer.zero_grad()
                disaggregation = self.model(main)

                for i in range(self.config['initialize']['device_num']):
                    ReconLoss += self.mse(disaggregation[i], target[i])
                MSELossTrain += ReconLoss.item()
                ReconLoss.backward()
                optimizer.step()

            MSELossTrain /= idx

            if epoch % 10 == 0:
                scheduler.step()

            if epoch % 1 == 0:
                maeAvg = self.MyEval(self.model, TestLoader)
                if MinMAE > np.mean(maeAvg):
                    MinMAE = np.mean(maeAvg)
                    torch.save(
                        self.model.state_dict(),
                        f'{self.checkpoint}/best_model.pth',
                    )

    def readHistoryData(self):
        allSample = []
        historyData = os.listdir(self.historyDataPath)
        for file in historyData:
            filePath = os.path.join(self.historyDataPath, file)
            data = loadJsonFile(Path(filePath))
            allSample.append([data['ch_4'], data['ch_5'], data['ch_6'], data['ch_7'], data['ch_8'], data['ch_9']])

        return allSample

    def createDataloader(self, allSample):
        TrainDataset = MyDataset(allSample)

        TrainSize = int(len(TrainDataset) * self.config['TrainPercentage'])
        TestSize = len(TrainDataset) - TrainSize
        trainDataset, testDataset = torch.utils.data.random_split(
            TrainDataset, [TrainSize, TestSize]
        )

        TrainLoader = DataLoader(
            trainDataset,
            batch_size=self.config['BatchSize'],
            shuffle=True,
            num_workers=0,
        )
        TestLoader = DataLoader(
            testDataset, batch_size=1, shuffle=False, num_workers=0
        )

        return TrainLoader, TestLoader

    def MyEval(self, model, TestLoader):
        res = []
        MetricMAE = [0] * self.config['initialize']['device_num']

        model.eval()
        with torch.no_grad():
            for idx, (main, target) in enumerate(TestLoader):
                main = main.type(torch.FloatTensor).to(device=self.device)
                target = target.type(torch.FloatTensor).to(device=self.device)
                target = target.transpose(0, 1)

                disaggregation = model(main)

                disaggregation *= 4000
                target *= 4000

                mae = self.GetMetric(disaggregation, target, 'MAE')  # 计算mae指标
                for i, item in enumerate(mae):
                    MetricMAE[i] += item

            maeAvg = np.array(MetricMAE) / idx

        return maeAvg

    def GetMetric(self, disaggregation, target, metric='MSE'):
        num = disaggregation.shape[0]
        MetricDevice = []
        for i in range(num):  # 逐设备计算指标
            MetricDevice.append(
                self.mse(disaggregation[i], target[i]).item()
                if metric == 'MSE'
                else self.mae(disaggregation[i], target[i]).item()
            )

        return MetricDevice
