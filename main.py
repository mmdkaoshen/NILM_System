import os
import queue
import time

import numpy as np
from flask import Flask, jsonify

from System.FetchData import FetchDataThread
from System.Disaggregator import DisaggregatorThread
from System.ResultProcessThread import ResultProcessThread
from System.ScheduledTask import ScheduledTask
from System.Trainer import Trainer
from System.Utilis import loadJsonFile, fakeData


app = Flask(__name__)
dataQueue = queue.Queue()
resultQueue = queue.Queue()
fakeData(r'D:\ProgramData\code\python\MossFormer\data\current\real\5mix\iPad_vivo_monitor_fan_pot\save,2023年10月21日15时21分01秒,1(是否同步),9CH,1(时间单位),1000.000Hz,单端,.dat', dataQueue)
sampleDataPath = r'D:\Data\SegData\Data'
config = loadJsonFile('./Model/NILMFormer/config/config.json')
fetchDataThread = FetchDataThread(sampleDataPath, dataQueue)
disaggregatorThread = DisaggregatorThread(config, dataQueue, resultQueue)
scheduledTask = ScheduledTask(disaggregatorThread.accumulatePower)
fetchDataThread.daemon = True
disaggregatorThread.daemon = True
scheduledTask.daemon = True


@app.route('/disaggregate')
def getDisaggregationRes():
    result = resultQueue.get()
    returnData = {
        'energy': result[0],
        'status': result[1],
        'disaggregation': result[2]
    }

    return jsonify(returnData)


@app.route('/')
def getRawData():
    rawData = dataQueue.get()
    returnData = {'voltage': rawData[1], 'current': rawData[2]}

    return jsonify(returnData)


@app.route('/train')
def trainModel():
    trainer = Trainer(config, sampleDataPath)
    trainer.start()


if __name__ == "__main__":
    if os.environ.get('WERKZEUG_RUN_MAIN'):
        disaggregatorThread.start()
        # fetchDataThread.start()
        scheduledTask.start()
    app.run(port=2020, host="127.0.0.1", debug=True)
