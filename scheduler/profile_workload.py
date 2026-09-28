import os
import torch

from torch.profiler import (
    profile,
    ProfilerActivity,
    ExecutionTraceObserver,
)

from pytorch_workload import create_workload


TRACE_DIR = "traces"
HOST_TRACE = os.path.join(TRACE_DIR, "pytorch_et_rank0.json")
DEVICE_TRACE = os.path.join(TRACE_DIR, "kineto_trace_rank0.json")


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)

    model, x = create_workload()
    model.eval()

    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        activities.append(ProfilerActivity.CUDA)

    print("PyTorch version:", torch.__version__)
    print("Profiling on:", x.device)
    print("Activities:", activities)

    et = ExecutionTraceObserver()
    et.register_callback(HOST_TRACE)

    # Older PyTorch compatibility:
    # manually start the execution trace observer instead of
    # passing execution_trace_observer= into profile().
    et.start()

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as prof:

        for step in range(3):
            with torch.no_grad():
                y = model(x)

            prof.step()

            print(
                f"step {step}: "
                f"input={tuple(x.shape)} "
                f"output={tuple(y.shape)}"
            )

    et.stop()
    et.unregister_callback()

    # Save the regular Kineto/Chrome profiler trace.
    prof.export_chrome_trace(DEVICE_TRACE)

    print("\nGenerated:")
    print(" ", HOST_TRACE)
    print(" ", DEVICE_TRACE)


if __name__ == "__main__":
    main()
