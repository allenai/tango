"""
Usage: python -m scripts.beaker_submit_gpu_tests NAME COMMIT_SHA
"""

import sys

import rich
from beaker import (
    EnvVar,
    ExperimentSpec,
    ImageSource,
    ResultSpec,
    TaskContext,
    TaskResources,
    TaskSpec,
)

from tango.common.exceptions import SigTermReceived
from tango.common.logging import cli_logger, initialize_logging
from tango.common.util import install_sigterm_handler


def main(name: str, commit_sha: str):
    initialize_logging(log_level="info", enable_cli_logs=True, console_width=180)
    install_sigterm_handler()

    from .beaker_common import (
        BEAKER_CLOUD_CLUSTER,
        BEAKER_IMAGE,
        BEAKER_ON_PREM_CLUSTERS,
        beaker,
    )

    cli_logger.info(f"Authenticated as {beaker.account.name}")

    # Find a cluster to use. We default to using our scalable cloud cluster,
    # but we'll also check to see if we can find an on-prem cluster with enough
    # free GPUs.
    beaker_cluster = BEAKER_CLOUD_CLUSTER
    for on_prem_cluster in BEAKER_ON_PREM_CLUSTERS:
        for node_util in beaker.cluster.utilization(on_prem_cluster):
            if node_util.free.gpu_count is not None and node_util.free.gpu_count >= 2:
                beaker_cluster = on_prem_cluster
                cli_logger.info(f"Found on-prem cluster with enough free GPUs ({on_prem_cluster})")
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

    cli_logger.info("Experiment spec: %s", spec.to_json())

    cli_logger.info("Submitting experiment...")
    experiment = beaker.experiment.create(name, spec)
    cli_logger.info(
        f"Experiment {experiment.id} submitted.\nSee progress at https://beaker.org/ex/{experiment.id}",
    )

    try:
        cli_logger.info("Waiting for job to finish...")
        experiment = beaker.experiment.await_all(experiment, timeout=20 * 60)

        cli_logger.info("Pulling logs...")
        logs = "".join([line.decode() for line in beaker.experiment.logs(experiment)])
        rich.get_console().rule("Logs")
        cli_logger.info(logs)

        sys.exit(experiment.jobs[0].status.exit_code)
    except (KeyboardInterrupt, SigTermReceived):
        cli_logger.info("Canceling job...")
        beaker.experiment.stop(experiment)


if __name__ == "__main__":
    main(*sys.argv[1:])
