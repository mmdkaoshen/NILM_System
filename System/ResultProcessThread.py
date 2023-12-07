import queue
import threading

import schedule

from System.AccumulatePower import AccumulatePower
from System.DetectSwitchStatus import DetectSwitchStatus
from System.ScheduledTask import ScheduledTask


class ResultProcessThread(threading.Thread):
    def __init__(self, resultQueue: queue.Queue):
        super().__init__()
        self.resultQueue = resultQueue
        self.resultData = []
        self.accumulatePower = AccumulatePower(5)
        self.detectSwitchStatus = DetectSwitchStatus(5)

    def run(self):
        schedule.every(1).seconds.do(self.processOnce)
        while True:
            schedule.run_pending()

    def getResultData(self):
        self.resultData = self.resultQueue.get()

    def processOnce(self):
        self.getResultData()
        self.accumulatePower.accumulatePower(self.resultData)
        self.detectSwitchStatus.getRealTimeSwitchStatus(self.resultData)
