import logging
import os
import tempfile
import time
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import jsonpickle
from beaker import (
    Beaker,
    DataMount,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Digest,
    EnvVar,
    ExperimentSpec,
    JobFailedError,
    TaskResources,
    TaskSpec,
)

from tango.common.exceptions import ConfigurationError, ExecutorError
from tango.executor import Executor, ExecutorOutput
from tango.step import Step
from tango.step_graph import StepGraph
from tango.version import VERSION
from tango.workspace import Workspace

logger = logging.getLogger(__name__)


@Executor.register("beaker")
class BeakerExecutor(Executor):
    """
    This is a :class:`~tango.executor.Executor` that runs steps on `Beaker`_.

    :param workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.

    .. tip::
        Registered as :class:`~tango.executor.Executor` under the name "beaker".
    """

    GITHUB_TOKEN_SECRET_NAME: str = "TANGO_GITHUB_TOKEN"

    RESULTS_DIR: str = "/tango/output"

    ENTRYPOINT_DIR: str = "/tango/entrypoint"

    ENTRYPOINT_FILENAME: str = "entrypoint.sh"

    INPUT_DIR: str = "/tango/input"

    STEP_GRAPH_FILENAME: str = "config.json"

    def __init__(
        self,
        workspace: Workspace,
        clusters: List[str],
        include_package: Optional[Sequence[str]] = None,
        beaker_workspace: Optional[str] = None,
        github_token: Optional[str] = None,
        beaker_image: Optional[str] = "ai2/conda",
        docker_image: Optional[str] = None,
        datasets: Optional[List[DataMount]] = None,
        env_vars: Optional[List[EnvVar]] = None,
        parallelism: Optional[int] = -1,
        **kwargs,
    ):
        # Pre-validate arguments.
        if (beaker_image is None) == (docker_image is None):
            raise ConfigurationError(
                "Either 'beaker_image' or 'docker_image' must be specified for BeakerExecutor, but not both."
            )

        super().__init__(workspace, include_package=include_package, parallelism=parallelism)
        self.beaker = Beaker.from_env(default_workspace=beaker_workspace, session=True)
        self.beaker_image = beaker_image
        self.docker_image = docker_image
        self.datasets = datasets
        self.env_vars = env_vars
        self.clusters = clusters

        try:
            self.github_token: str = github_token or os.environ["GITHUB_TOKEN"]
        except KeyError:
            raise ConfigurationError(
                "A GitHub personal access token with the 'repo' scope is required. "
                "This can be set with the 'github_token' argument to the BeakerExecutor, "
                "or as the environment variable 'GITHUB_TOKEN'."
            )

    def execute_step(self, step: Step) -> None:
        raise NotImplementedError

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        import concurrent.futures

        steps_to_run: Set[str] = set()
        successful: Set[str] = set()
        failed: Set[str] = set()
        not_run: Set[str] = set()
        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()

        def update_steps_to_run():
            for step_name, step in step_graph.items():
                if (
                    step.unique_id in successful
                    or step.unique_id in failed
                    or step.unique_id in not_run
                ):
                    # Make sure step is no longer in queue.
                    steps_to_run.discard(step_name)  # This does NOT raise KeyError if not found
                else:
                    # Check dependencies.
                    for dependency in step.dependencies:
                        if dependency.unique_id not in successful and dependency.cache_results:
                            if dependency.unique_id in failed or dependency.unique_id in not_run:
                                # A dependency failed or can't be run, so this step can't be run.
                                not_run.add(step.unique_id)
                            break
                    else:
                        # Dependencies are OK, so we can run this step now.
                        if step.cache_results or step in uncacheable_leaf_steps:
                            steps_to_run.add(step_name)

        def make_future_done_callback(step_name: str):
            def future_done_callback(future: concurrent.futures.Future):
                step_id = step_graph[step_name].unique_id
                try:
                    if future.exception() is None:
                        successful.add(step_id)
                    else:
                        failed.add(step_id)
                except concurrent.futures.TimeoutError:
                    failed.add(step_id)

        update_steps_to_run()

        step_futures: List[concurrent.futures.Future] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallelism or None) as pool:
            while steps_to_run:
                # Submit steps left to run.
                for step_name in steps_to_run:
                    future = pool.submit(
                        self.execute_sub_graph_for_step, step_graph, step_name, run_name
                    )
                    future.add_done_callback(make_future_done_callback(step_name))
                    step_futures.append(future)

                # Wait for something to happen.
                _, not_done = concurrent.futures.wait(
                    step_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                # Update the list of running futures.
                step_futures.clear()
                step_futures = list(not_done)

                # Update the steps we still have to run.
                update_steps_to_run()

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)

    def execute_sub_graph_for_step(
        self, step_graph: StepGraph, step_name: str, run_name: Optional[str] = None
    ) -> None:
        step = step_graph[step_name]

        # Initialize experiment and task spec.
        spec = self._build_experiment_spec(step_graph, step_name)

        # Create experiment.
        experiment_name = f"{step.unique_id}-{str(uuid.uuid4())}"
        experiment = self.beaker.experiment.create(experiment_name, spec)
        logger.info(
            "Submitted Beaker experiment %s for step '%s'",
            self.beaker.experiment.url(experiment),
            step.unique_id,
        )

        # Follow the experiment and stream the logs until it completes.
        setup_stage: bool = True
        try:
            for line in self.beaker.experiment.follow(experiment, strict=True):
                # Every log line from Beaker starts with an RFC 3339 UTC timestamp
                # (e.g. '2021-12-07T19:30:24.637600011Z'). We don't want to print
                # the timestamps so we split them off like this:
                line = line[line.find(b"Z ") + 2 :]
                line_str = line.decode(errors="ignore").rstrip()

                # Try parsing a JSON log record from the line.
                try:
                    log_record_attrs = jsonpickle.loads(line_str)
                    log_record = logging.makeLogRecord(log_record_attrs)
                    setup_stage = False
                    logging.getLogger(log_record.name).handle(log_record)
                except JSONDecodeError:
                    if setup_stage:
                        logger.debug(f"[step {step_name}, setup] {line_str}")
                    else:
                        logger.info(f"[step {step_name}] {line_str}")
        except JobFailedError:
            raise ExecutorError(
                f"Beaker job for step '{step_name}' failed. "
                f"You can check the logs at {self.beaker.experiment.url(experiment)}"
            )

    @staticmethod
    def _parse_git_remote(url: str) -> Tuple[str, str]:
        """
        Parse a git remote URL into a GitHub (account, repo) pair.
        """
        account, repo = (
            url.split("https://github.com/")[-1]
            .split("git@github.com:")[-1]
            .split(".git")[0]
            .split("/")
        )
        return account, repo

    def _ensure_entrypoint_dataset(self) -> Dataset:
        import hashlib
        from importlib.resources import read_binary

        import tango.integrations.beaker

        workspace_id = self.beaker.workspace.get().id

        # Get hash of the local entrypoint source file.
        sha256_hash = hashlib.sha256()
        contents = read_binary(tango.integrations.beaker, "entrypoint.sh")
        sha256_hash.update(contents)

        entrypoint_dataset_name = f"tango-v{VERSION}-{workspace_id}-{sha256_hash.hexdigest()[:6]}"

        # Ensure entrypoint dataset exists.
        entrypoint_dataset: Dataset
        try:
            entrypoint_dataset = self.beaker.dataset.get(entrypoint_dataset_name)
        except DatasetNotFound:
            # Create it.
            logger.debug(f"Creating entrypoint dataset '{entrypoint_dataset_name}'")
            try:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdir = Path(tmpdirname)
                    entrypoint_path = tmpdir / "entrypoint.sh"
                    with open(entrypoint_path, "wb") as entrypoint_file:
                        entrypoint_file.write(contents)
                    entrypoint_dataset = self.beaker.dataset.create(
                        entrypoint_dataset_name, entrypoint_path
                    )
            except DatasetConflict:  # could be in a race with another `tango` process.
                time.sleep(1.0)
                entrypoint_dataset = self.beaker.dataset.get(entrypoint_dataset_name)

        # Verify contents.
        err_msg = (
            f"Checksum failed for entrypoint dataset {self.beaker.dataset.url(entrypoint_dataset)}\n"
            f"This could be a bug, or it could mean someone has tampered with the dataset.\n"
            f"If you're sure no one has tampered with it, you can delete the dataset from "
            f"the Beaker dashboard and try again."
        )
        ds_files = list(self.beaker.dataset.ls(entrypoint_dataset))
        if len(ds_files) != 1:
            raise ExecutorError(err_msg)
        if ds_files[0].digest != Digest(sha256_hash.digest()):
            raise ExecutorError(err_msg)

        return entrypoint_dataset

    def _ensure_step_graph_dataset(self, step_graph: StepGraph) -> Dataset:
        step_graph_dataset_name = f"tango-v{VERSION}-{str(uuid.uuid4())}"
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname)
                path = tmpdir / self.STEP_GRAPH_FILENAME
                step_graph.to_file(path, include_unique_id=True)
                dataset = self.beaker.dataset.create(step_graph_dataset_name, path)
        except DatasetConflict:  # could be in a race with another `tango` process.
            time.sleep(1.0)
            dataset = self.beaker.dataset.get(step_graph_dataset_name)
        return dataset

    def _ensure_cluster(self, task_resources: TaskResources) -> str:
        cluster_to_use: str
        if not self.clusters:
            raise ConfigurationError("At least one cluster is required in 'clusters'")
        elif len(self.clusters) == 1:
            cluster_to_use = self.clusters[0]
        else:
            available_clusters = sorted(
                self.beaker.cluster.filter_available(task_resources, *self.clusters),
                key=lambda x: x.queued_jobs,
            )
            if available_clusters:
                cluster_to_use = available_clusters[0].cluster.full_name
                logger.debug(f"Using cluster '{cluster_to_use}'")
            else:
                cluster_to_use = self.clusters[0]
                logger.debug(
                    "No clusters currently have enough free resources available. "
                    "Will use '%' anyway.",
                    cluster_to_use,
                )
        return cluster_to_use

    def _build_experiment_spec(self, step_graph: StepGraph, step_name: str) -> ExperimentSpec:
        step = step_graph[step_name]
        sub_graph = step_graph.sub_graph(step_name)
        step_info = self.workspace.step_info(step)

        step_resources = step.resources
        task_resources = TaskResources(
            cpu_count=step_resources.cpu_count,
            gpu_count=step_resources.gpu_count,
            memory=step_resources.memory,
            shared_memory=step_resources.shared_memory,
        )

        # Ensure we're working in a GitHub repository.
        if (
            step_info.environment.git is None
            or step_info.environment.git.commit is None
            or step_info.environment.git.remote is None
            or "github.com" not in step_info.environment.git.remote
        ):
            raise ExecutorError(
                f"Missing git data for step '{step.unique_id}'. "
                f"BeakerExecutor requires a git repository with a GitHub remote."
            )
        try:
            github_account, github_repo = self._parse_git_remote(step_info.environment.git.remote)
        except ValueError:
            raise ExecutorError("BeakerExecutor requires a git repository with a GitHub remote.")
        git_ref = step_info.environment.git.commit

        # Ensure dataset with the entrypoint script exists and get it.
        entrypoint_dataset = self._ensure_entrypoint_dataset()

        # Create dataset for step graph.
        step_graph_dataset = self._ensure_step_graph_dataset(sub_graph)

        # Write the GitHub token secret.
        self.beaker.secret.write(self.GITHUB_TOKEN_SECRET_NAME, self.github_token)

        # Build Tango command to run.
        command = [
            "tango",
            "--log-level",
            "debug",
            "--called-by-executor",
            "beaker-executor-run",
            self.INPUT_DIR + "/" + self.STEP_GRAPH_FILENAME,
            step.name,
            self.workspace.url,
        ]
        if self.include_package is not None:
            for package in self.include_package:
                command += ["-i", package]

        # Get cluster to use.
        cluster = self._ensure_cluster(task_resources)

        # Build task spec.
        task_spec = (
            TaskSpec.new(
                step.unique_id,
                cluster,
                beaker_image=self.beaker_image,
                docker_image=self.docker_image,
                result_path=self.RESULTS_DIR,
                command=["bash", self.ENTRYPOINT_DIR + "/" + self.ENTRYPOINT_FILENAME],
                arguments=command,
                resources=task_resources,
                datasets=self.datasets,
                env_vars=self.env_vars,
            )
            .with_env_var(name="TANGO_VERSION", value=VERSION)
            .with_env_var(name="GITHUB_TOKEN", secret=self.GITHUB_TOKEN_SECRET_NAME)
            .with_env_var(name="GITHUB_REPO", value=f"{github_account}/{github_repo}")
            .with_env_var(name="GIT_REF", value=git_ref)
            .with_env_var(name="PYTHON_VERSION", value=step_info.environment.python)
            .with_dataset(self.ENTRYPOINT_DIR, beaker=entrypoint_dataset.id)
            .with_dataset(self.INPUT_DIR, beaker=step_graph_dataset.id)
        )

        return ExperimentSpec(tasks=[task_spec])
