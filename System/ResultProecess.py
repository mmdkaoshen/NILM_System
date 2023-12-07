import threading
from queue import Queue

import schedule


class ResultProcess(threading.Thread):
    def __init__(self, resultQueue: Queue, deviceNum: int):
        self.resultQueue = resultQueue
        self.resultData = []

        self.accumulatedPowerInterval = 0
        self.periodPowerStartTime = 0.0
        self.periodPowerEndTime = 0.0
        self.periodPower = [0] * deviceNum
        self.totalPower = [0] * deviceNum

        self.allStatus = [0] * deviceNum

    def getResult(self):
        if not self.resultQueue.empty():
            self.resultData = self.resultQueue.get(block=True, timeout=5)

    def processOnce(self):
        self.getResult()
        self.accumulateTotalPower()
        self.accumulatePeriodPower()

    def run(self) -> None:
        job = schedule.every(1).seconds.do(self.processOnce)
        while True:
            schedule.run_pending()
