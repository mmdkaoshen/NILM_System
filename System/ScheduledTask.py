import threading
import datetime
import time

import pymysql

from System.AccumulatePower import AccumulatePower


class ScheduledTask(threading.Thread):
    def __init__(self, accumulatePower: AccumulatePower):
        super().__init__()
        self.connection = pymysql.Connect(
            host="localhost",
            user="root",
            password="123456"
        )
        self.accumulatePower = accumulatePower
        self.preEnergy = [0] * self.accumulatePower.deviceNum

    def task(self, periodPowerStartTime, periodPowerEndTime):
        periodEnergy = self.calculateEnergy()

        queryUse = "use powermonitoring;"
        queryInsert = "insert into device_energy(device_id, start_time, end_time, energy) value(%s, %s, %s, %s);"

        self.connection.ping()
        with self.connection.cursor() as cursor:
            cursor.execute(queryUse)
            for i in range(self.accumulatePower.deviceNum):
                cursor.execute(queryInsert,
                               (i, periodPowerStartTime, periodPowerEndTime, periodEnergy[i]))
            self.connection.commit()

    def calculateEnergy(self):
        currentEnergy = self.accumulatePower.totalPower
        periodEnergy = [x - y for x, y in zip(currentEnergy, self.preEnergy)]
        self.preEnergy = currentEnergy

        return periodEnergy

    def run(self):
        while True:
            # 获取当前时间
            now = datetime.datetime.now()

            # 设置下一个整点的时间
            nextHour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            # 设置上一个整点的时间
            preHour = (nextHour - datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            nextHourStr = nextHour.strftime('%Y-%m-%d %H:%M:%S')
            preHourStr = preHour.strftime('%Y-%m-%d %H:%M:%S')

            # 计算等待时间
            waitTime = (nextHour - now).total_seconds()

            # 等待到下一个整点
            time.sleep(waitTime)

            # 执行任务
            self.task(preHourStr, nextHourStr)
