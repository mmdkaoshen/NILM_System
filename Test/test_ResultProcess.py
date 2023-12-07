import queue
from datetime import datetime

import pytest

from System.ResultProecess import ResultProcess


class TestResultProcess:
    @pytest.mark.parametrize(
        "resultQueue, deviceNum",
        [
            (queue.Queue(), 5),
        ]
    )
    def test_tc01(self, resultQueue: queue.Queue, deviceNum: int):
        for _ in range(19):
            now = datetime.now()
            timestamp = now.timestamp()
            resultQueue.put([timestamp, [1] * deviceNum, [1] * deviceNum], block=True, timeout=5)
        resultProcess = ResultProcess(resultQueue, deviceNum)
        resultProcess.run()

        assert resultProcess.totalPower == [19] * deviceNum
        assert resultProcess.periodPower == [9] * deviceNum
