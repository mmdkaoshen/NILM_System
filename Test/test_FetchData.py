import os
import queue
from datetime import datetime
from pathlib import Path

import pytest

from System.FetchData import loadJsonFile, FetchDataThread


class TestFetchData:
    @pytest.mark.parametrize(
        "dirPath, dataQueue",
        [
            (Path(r'D:\Data\SegData\Data'), queue.Queue()),
        ]
    )
    def test_tc01(self, dirPath: Path, dataQueue: queue.Queue):
        testFetchDataThread = FetchDataThread(dirPath, dataQueue)
        testFetchDataThread.fetchData()
        targetData = loadJsonFile(r'D:\Data\SegData\Data\2023年10月30日 19时24分28秒.json')
        testData = testFetchDataThread.dataQueue.get()

        assert targetData['ch_1'][:1000] == testData[1] and targetData['ch_4'][:1000] == testData[2]
