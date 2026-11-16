from typing import override

from schedulers.prio_heap_on_fifo import PrioHeapOnFIFO
from schedulers.task import Task, TaskList


class LongestJobFirst(PrioHeapOnFIFO):
    @override
    def admit(self, tasks: TaskList):
        if isinstance(tasks, Task):
            tasks = [tasks]

        for task in tasks:
            duration = task.event.duration_us
            task.priority = -(duration or 0)

        super().admit(tasks)
