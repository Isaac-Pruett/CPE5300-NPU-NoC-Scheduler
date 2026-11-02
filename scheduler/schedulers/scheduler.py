from abc import ABC, abstractmethod

from schedulers.task import Task, TaskList


class Scheduler(ABC):
    def __init__(self):
        self.queue: list[Task] = []  # if TaskList, then its MLQ scheduling

    @abstractmethod
    def next(self) -> Task:
        pass

    def admit(self, tasks: TaskList):
        if isinstance(tasks, Task):
            tasks = [tasks]
        self.queue.extend(tasks)
