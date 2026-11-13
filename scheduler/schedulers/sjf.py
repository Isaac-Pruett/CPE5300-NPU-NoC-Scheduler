from typing import override

from schedulers.prio_heap_on_fifo import PrioHeapOnFIFO
from schedulers.task import Task, TaskList


class ShortestJobFirst(PrioHeapOnFIFO):
    @override
    def admit(self, tasks: TaskList):
        if isinstance(tasks, Task):
            tasks = [tasks]

        for t in tasks:
            t.priority = t.event.duration_us

        super().admit(tasks)
