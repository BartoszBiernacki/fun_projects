from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, current_process
import random
from time import sleep


class Data:
    def __init__(self):
        pass

    def generate(self, queue: Queue):
        for i in range(5):
            sleep(random.randint(1, 2))
            queue.put(i)
            print(f"{i} was putted")
        queue.put(None)

    def handle_data(self, queue: Queue):
        while True:
            item = queue.get()
            if item is not None:
                print(f"{item} is processed by {current_process().name} ...")
                sleep(random.randint(1, 5))
                print(f"{item} was processed by {current_process().name} ...")
            else:
                queue.put(None)
                exit()

    def run(self):
        queue = Queue()
        processes = []

        generator = Process(target=self.generate, args=(queue,))
        processes.append(generator)

        for _ in range(2):
            handler = Process(target=self.handle_data, args=(queue,))
            processes.append(handler)

        [x.start() for x in processes]


class Data2:
    def __init__(self):
        pass

    def generate(self, in_queue: Queue, out_queue: Queue):
        while not in_queue.empty():
            in_item = in_queue.get()
            sleep(in_item)
            val = in_item**2
            out_queue.put(val)
            print(f"{val} was putted by {current_process().name}")

        out_queue.put(None)

    def handle_data(self, queue: Queue):
        while True:
            item = queue.get()
            if item is not None:
                print(f"{item} is processed by {current_process().name} ...")
                sleep(random.randint(1, 5))
                print(f"{item} was processed by {current_process().name} ...")
            else:
                queue.put(None)
                print('breaking')
                break

    def run(self):
        input_queue, output_queue = Queue(), Queue()
        processes = []

        for i in range(6):
            input_queue.put(i)

        for _ in range(2):
            generator = Process(target=self.generate, args=(input_queue,
                                                            output_queue))
            processes.append(generator)

        for _ in range(5):
            handler = Process(target=self.handle_data, args=(output_queue,))
            processes.append(handler)

        [x.start() for x in processes]


if __name__ == '__main__':
    data2 = Data2()
    data2.run()

