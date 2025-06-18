import time
from datetime import datetime

class TimeMeasuring:
    def __init__(self, filename="time_log"):
        self.__time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = filename + timestamp + ".txt"

        self.log = []

    def took(self, name):
        start = self.__time
        self.__time = time.time()
        elapsed_time = (self.__time - start) * 1000
        log_entry = f"{name} needed {elapsed_time:.5f} ms"

        self.log.append(log_entry)
        print(log_entry)

    def reset(self):
        self.__time = time.time()

    def save_log(self):
        with open(self.filename, "a") as f:
            for entry in self.log:
                f.write(entry + "\n")
        print(f"Logs saved to {self.filename}")