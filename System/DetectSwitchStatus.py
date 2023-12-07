import datetime

import numpy as np
import pymysql

from System.Utilis import timeStr


class DetectSwitchStatus:
    def __init__(self, deviceNum: int):
        self.resultData = []
        self.open = [False] * deviceNum
        self.switchOnTime = np.zeros((deviceNum,))
        self.switchOffTime = np.zeros((deviceNum,))
        self.preSwitchStatus = np.zeros((deviceNum,), dtype=int)

    def getRealTimeSwitchStatus(self, resultData: list):
        self.resultData = resultData

    def switchOnInterval(self, timestamp, status):
        onIdx = np.where(np.array(status) == 1)[0]
        realIdx = np.where(self.preSwitchStatus[onIdx] != 1)[0]
        self.switchOnTime[onIdx[realIdx]] = timestamp
        self.preSwitchStatus = np.array(status)

        offIdx = np.where(np.array(status) == -1)[0]
        if len(offIdx) != 0:
            self.switchOffTime[offIdx] = timestamp

            self.saveToDatabase(offIdx)

    def saveToDatabase(self, offIdx):
        connection = pymysql.Connect(
            host="localhost",
            user="root",
            password="123456"
        )

        queryUse = "use switchstatus;"
        queryInsert = "insert into switch_status(device_id, start_time, end_time) value(%s, %s, %s);"

        with connection.cursor() as cursor:
            cursor.execute(queryUse)
            for i in offIdx:
                cursor.execute(queryInsert,
                               (i, timeStr(self.switchOnTime[i]), timeStr(self.switchOffTime[i])))
            connection.commit()
