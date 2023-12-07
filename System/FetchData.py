from datetime import datetime
import os
import threading
from pathlib import Path
from queue import Queue

import schedule

from System.Utilis import loadJsonFile


class FetchDataThread(threading.Thread):
    def __init__(self, dirPath, dataQueue: Queue):
        super().__init__()
        self.dirPath = dirPath
        self.preFile = ''
        self.dataQueue = dataQueue
        self.voltage = []
        self.current = []

    def run(self):
        schedule.every(1).seconds.do(self.fetchData)
        while True:
            schedule.run_pending()

    def fetchData(self):
        allFile = os.listdir(self.dirPath)
        if len(allFile) != 0:
            allFile.sort(
                key=lambda x: int(datetime.strptime(x[:-5], "%Y年%m月%d日 %H时%M分%S秒").strftime("%Y%m%d%H%M%S")))
            filePath = os.path.join(self.dirPath, allFile[-1])
            if self.preFile != filePath:
                data = loadJsonFile(Path(filePath))
                now = datetime.now()
                timestamp = now.timestamp()
                self.preFile = filePath
                self.dataQueue.put([timestamp, data['ch_1'][:1000], data['ch_4'][:1000]], block=True, timeout=5)
                # print("read one file")

    def fetchDataTest(self):
        allFile = os.listdir(self.dirPath)
        if len(allFile) != 0:
            allFile.sort(
                key=lambda x: int(datetime.strptime(x[:-5], "%Y年%m月%d日 %H时%M分%S秒").strftime("%Y%m%d%H%M%S")))
            filePath = os.path.join(self.dirPath, allFile[-1])
            data = loadJsonFile(Path(filePath))
            now = datetime.now()
            timestamp = now.timestamp()
            self.dataQueue.put([timestamp, data['ch_1'][:1000], data['ch_4'][:1000]], block=True)
