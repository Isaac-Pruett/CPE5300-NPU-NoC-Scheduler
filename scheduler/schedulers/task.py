import dataclasses

type TaskList = Task | list[Task]


# TODO fill in parameters and add other modes
@dataclasses.dataclass
class Task:
    latency: int
    preemptible: bool
    priority: int | None = None

    def __lt__(self, other: "Task") -> bool:
        if self.priority is not None and other.priority is not None:
            if self.priority < other.priority:
                return True

        return False
