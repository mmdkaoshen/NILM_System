import datetime
import json
import queue
from pathlib import Path
import numpy as np


def loadJsonFile(filePath: Path) -> dict:
    f = open(filePath, 'r')
    content = f.read()

    return json.loads(content)


def timeStr(timestamp):
    dateTime = datetime.datetime.fromtimestamp(timestamp)

    return dateTime.strftime("%Y-%m-%d %H:%M:%S")


def readDatFile(filePath):
    data = np.fromfile(filePath, dtype='>f')
    data = data.reshape(-1, 9)
    voltage = data[:1314000, 0]
    mains = data[:1314000, 3]
    voltage = voltage.reshape(-1, 1000)
    mains = mains.reshape(-1, 1000)

    return voltage.tolist(), mains.tolist()


def pushToQueue(dataQueue: queue.Queue, timestamp, voltage, current):
    dataQueue.put([timestamp, voltage, current])


def fakeData(filePath, dataQueue: queue.Queue):
    voltage, current = readDatFile(filePath)
    for i, j in zip(voltage, current):
        nowTime = datetime.datetime.now().timestamp()
        pushToQueue(dataQueue, nowTime, i, j)


def realStatus(preStatus, curStatus):
    xorRes = [x ^ y for x, y in zip(preStatus, curStatus)]
    for i, elem in enumerate(xorRes):
        if elem == 1:
            preStatus[i] = 1
        elif elem == -2:
            preStatus[i] = -1
        else:
            preStatus[i] = 0
