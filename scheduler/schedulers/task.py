import dataclasses

from speculative import Event

type TaskList = Task | list[Task]


# TODO fill in parameters and add other modes
@dataclasses.dataclass
class Task:
    event: Event
    preemptible: bool
    priority: int | None = None
    finished: bool = False

    def __lt__(self, other: "Task") -> bool:
        if self.priority is not None and other.priority is not None:  # noqa: SIM102
            if self.priority < other.priority:
                return True

        return False
