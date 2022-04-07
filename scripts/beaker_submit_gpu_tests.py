"""
Usage: python -m scripts.beaker_submit_gpu_tests NAME COMMIT_SHA
"""

import signal
import sys

from rich import pretty, print

pretty.install()

from beaker import (  # noqa: E402
    EnvVar,
    ExperimentSpec,
    ImageSource,
    ResultSpec,
    TaskContext,
    TaskResources,
    TaskSpec,
)

from .beaker_common import (  # noqa: E402
    BEAKER_CLOUD_CLUSTER,
    BEAKER_IMAGE,
    BEAKER_ON_PREM_CLUSTERS,
    beaker,
)


class TermInterrupt(Exception):
    pass


def handle_sigterm(sig, frame):
    raise TermInterrupt


def main(name: str, commit_sha: str):
    signal.signal(signal.SIGTERM, handle_sigterm)

    print(f"Authenticated as {beaker.account.name}")

    # Find a cluster to use. We default to using our scalable cloud cluster,
    # but we'll also check to see if we can find an on-prem cluster with enough
    # free GPUs.
    beaker_cluster = BEAKER_CLOUD_CLUSTER
    for on_prem_cluster in BEAKER_ON_PREM_CLUSTERS:
        for node_util in beaker.cluster.utilization(on_prem_cluster):
            if node_util.free.gpu_count is not None and node_util.free.gpu_count >= 2:
                beaker_cluster = on_prem_cluster
                print(f"Found on-prem cluster with enough free GPUs ({on_prem_cluster})")
                break
        else:
            continue
        break

    spec = ExperimentSpec(
        description="GPU Tests",
        tasks=[
            TaskSpec(
                name="tests",
                image=ImageSource(beaker=BEAKER_IMAGE),
                context=TaskContext(cluster=beaker_cluster),
                result=ResultSpec(path="/unused"),
                resources=TaskResources(gpu_count=2),
                env_vars=[EnvVar(name="COMMIT_SHA", value=commit_sha)],
                command=["/entrypoint.sh", "pytest", "-v", "-m", "gpu", "tests/"],
            )
        ],
    )

    print("Experiment spec:", spec)

    print("Submitting experiment...")
    experiment = beaker.experiment.create(name, spec)
    print(
        f"Experiment {experiment.id} submitted.\nSee progress at https://beaker.org/ex/{experiment.id}"
    )

    try:
        print("Waiting for job to finish...")
        experiment = beaker.experiment.await_all(experiment, timeout=20 * 60)

        print("Pulling logs...")
        logs = "".join([line.decode() for line in beaker.experiment.logs(experiment)])
        print(logs)

        sys.exit(experiment.jobs[0].status.exit_code)
    except (KeyboardInterrupt, TermInterrupt):
        print("Canceling job...")
        beaker.experiment.stop(experiment)


if __name__ == "__main__":
    main(*sys.argv[1:])
