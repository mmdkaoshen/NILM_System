import pymysql


class AccumulatePower:
    def __init__(self, deviceNum: int):
        self.deviceNum = deviceNum
        self.periodPower = [0] * deviceNum
        self.totalPower = [0] * deviceNum

    def accumulatePower(self, powerData: list):
        self.totalPower = [sum(element) for element in zip(self.totalPower, powerData)]

    def saveToDatabase(self):
        connection = pymysql.Connect(
            host="localhost",
            user="root",
            password="123456"
        )

        queryUse = "use powermonitoring;"
        queryInsert = "insert into device_energy(device_id, start_time, end_time, energy) value(%s, %s, %s, %s);"

        with connection.cursor() as cursor:
            cursor.execute(queryUse)
            for i in range(self.deviceNum):
                cursor.execute(queryInsert,
                               (i, self.periodPowerStartTime, self.periodPowerEndTime, self.periodPower[i]))
            connection.commit()
