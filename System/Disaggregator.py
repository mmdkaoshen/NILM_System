import os.path
import threading
from queue import Queue

import numpy as np
import pymysql
import schedule
from loguru import logger

import torch

from Model.NILMFormer.mossformer import MossFormer
from System.DataProcess import getPowerConsumption, detectSwitchStatus
from System.AccumulatePower import AccumulatePower
from System.DetectSwitchStatus import DetectSwitchStatus
from System.Utilis import realStatus


class DisaggregatorThread(threading.Thread):
    def __init__(self, config: dict, dataQueue: Queue, resultQueue: Queue):
        super().__init__()
        self.config = config
        self.dataQueue = dataQueue
        self.resultQueue = resultQueue
        self.model = None
        self.timestamp = 0
        self.voltage = []
        self.current = []
        self.disaggregation = []
        self.preStatus = [0] * config['initialize']['device_num']
        self.accumulatePower = AccumulatePower(config['initialize']['device_num'])
        self.detectSwitchStatus = DetectSwitchStatus(config['initialize']['device_num'])

        self.connection = pymysql.Connect(
            host="localhost",
            user="root",
            password="123456"
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            self.device = torch.device("cuda:0")
        else:
            torch.manual_seed(42)
            self.device = torch.device("cpu")

    def run(self):
        self.loadModel()

        schedule.every(1).seconds.do(self.processOnce)
        while True:
            schedule.run_pending()

    def loadModel(self):
        logger.info('Start loading model...')
        self.model = MossFormer(self.config['initialize'])
        self.model.to(self.device)
        logger.info('Model loading complete.')

        if not os.path.exists(self.config['weights']['path']):
            logger.error('Loading wights failed! Please download the weights file and put it in the correct path.')
        logger.info('Start loading pre-trained weights...')
        self.model.load_state_dict(torch.load(self.config['weights']['path'], map_location=self.device))
        logger.info('Weights loading complete.')

    def aggregate(self):
        mains = torch.Tensor(self.current).type(torch.FloatTensor).to(device=self.device)
        mains = mains.unsqueeze(0)
        mains /= 4000

        self.model.eval()
        with torch.no_grad():
            disaggregation = self.model(mains)

        disaggregation *= 4000
        disaggregation = disaggregation.squeeze(1)
        disaggregation = disaggregation.detach().cpu().numpy()
        disaggregation /= 100
        self.disaggregation = disaggregation.tolist()

        return disaggregation

    def calculateResult(self, current: np.ndarray):
        num = current.shape[0]
        power = []
        status = []
        for i in range(num):
            power.append(getPowerConsumption(self.voltage, list(current[i])))
            status.append(detectSwitchStatus(i, list(current[i])))

        self.accumulatePower.accumulatePower(power)
        realStatus(self.preStatus, status)
        self.detectSwitchStatus.switchOnInterval(self.timestamp, self.preStatus)

        self.resultQueue.put((self.accumulatePower.totalPower, status, current.tolist()), block=True, timeout=5)

    def getRawData(self):
        rawData = self.dataQueue.get(block=True)
        self.timestamp = rawData[0]
        self.voltage = list(map(lambda x: x / 1000 * 61.5, rawData[1]))
        self.current = rawData[2]

    def processOnce(self):
        if not self.dataQueue.empty():
            self.getRawData()
            self.calculateResult(self.aggregate())
