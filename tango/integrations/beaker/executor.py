import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    TaskStoppedError,
)
from git import Git, GitCommandError, InvalidGitRepositoryError, Repo

from tango.common.exceptions import (
    CancellationError,
    ConfigurationError,
    ExecutorError,
    RunCancelled,
)
from tango.common.logging import cli_logger, log_exception
from tango.executor import ExecutionMetadata, Executor, ExecutorOutput
from tango.step_graph import StepGraph
from tango.step_info import GitMetadata
from tango.version import VERSION
from tango.workspace import Workspace

from .common import Constants

logger = logging.getLogger(__name__)


class StepFailedError(ExecutorError):
    def __init__(self, msg: str, experiment_url: str):
        super().__init__(msg)
        self.experiment_url = experiment_url


@Executor.register("beaker")
class BeakerExecutor(Executor):
    """
    This is a :class:`~tango.executor.Executor` that runs steps on `Beaker`_.
    Each step is run as its own Beaker experiment.

    .. tip::
        Registered as an :class:`~tango.executor.Executor` under the name "beaker".

    .. important::
        The :class:`BeakerExecutor` requires that you run Tango within a GitHub repository and you push
        all of your changes prior to each ``tango run`` call. It also requires that you have
        a `GitHub personal access token <https://github.com/settings/tokens/new>`_
        with at least the "repo" scope set to the environment variable ``GITHUB_TOKEN``
        (you can also set it using the ``github_token`` parameter, see below).

        This is because :class:`BeakerExecutor` has to be able to clone your code from Beaker.

    .. important::
        The :class:`BeakerExecutor` will try to recreate your Python environment on Beaker
        every time a step is run, so it's important that you specify all of your dependencies
        in a PIP ``requirements.txt`` file, ``setup.py`` file, or a conda ``environment.yml`` file.
        Alternatively you could provide the ``install_cmd`` argument.

    .. important::
        The :class:`BeakerExecutor` takes no responsibility for saving the results of steps that
        it runs on Beaker. That's the job of your workspace. So make sure your using the
        right type of workspace or your results will be lost.

        For example, any "remote" workspace (like the :class:`BeakerWorkspace`) would work,
        or in some cases you could use a :class:`~tango.workspaces.LocalWorkspace` on an NFS drive.

    .. important::
        If you're running a step that requires special hardware, e.g. a GPU, you should
        specify that in the ``step_resources`` parameter to the step, or by overriding
        the step's :meth:`.resources() <tango.step.Step.resources>` property method.

    :param workspace: The :class:`~tango.workspace.Workspace` to use.
    :param clusters: A list of Beaker clusters that the executor may use to run steps.
    :param include_package: A list of Python packages to import before running steps.
    :param beaker_workspace: The name or ID of the Beaker workspace to use.
    :param github_token: You can use this parameter to set a GitHub personal access token instead of using
        the ``GITHUB_TOKEN`` environment variable.
    :param beaker_image: The name or ID of a Beaker image to use for running steps on Beaker.
        The image must come with bash and `conda <https://docs.conda.io/en/latest/index.html>`_
        installed (Miniconda is okay).
        This is mutually exclusive with the ``docker_image`` parameter. If neither ``beaker_image``
        nor ``docker_image`` is specified, the :data:`DEFAULT_BEAKER_IMAGE` will be used.
    :param docker_image: The name of a publicly-available Docker image to use for running
        steps on Beaker. The image must come with bash and `conda <https://docs.conda.io/en/latest/index.html>`_
        installed (Miniconda is okay).
        This is mutually exclusive with the ``beaker_image`` parameter.
    :param datasets: External data sources to mount into the Beaker job for each step. You could use
        this to mount an NFS drive, for example.
    :param env_vars: Environment variables to set in the Beaker job for each step.
    :param venv_name: The name of the conda virtual environment to use or create on the image.
        If you're using your own image that already has a conda environment you want to use,
        you should set this variable to the name of that environment.
        You can also set this to "base" to use the base environment.
    :param parallelism: Control the maximum number of steps run in parallel on Beaker.
    :param install_cmd: Override the command used to install your code and its dependencies
        in each Beaker job.
        For example, you could set ``install_cmd="pip install .[dev]"``.
    :param priority: The default task priority to assign to jobs ran on Beaker.
    :param allow_dirty: By default, the Beaker Executor requires that your git working directory has no uncommitted
        changes. If you set this to ``True``, we skip this check.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.

    .. attention::
        Certain parameters should not be included in the :data:`~tango.settings.TangoGlobalSettings.executor`
        part of your ``tango.yml`` file, namely ``workspace`` and ``include_package``.
        Instead use the top-level :data:`~tango.settings.TangoGlobalSettings.workspace`
        and :data:`~tango.settings.TangoGlobalSettings.include_package` fields, respectively.

    :examples:

    **Minimal tango.yaml file**

    You can use this executor by specifying it in your ``tango.yml`` settings file:

    .. code:: yaml

        executor:
          type: beaker
          beaker_workspace: ai2/my-workspace
          clusters:
            - ai2/general-cirrascale

    **Using GPUs**

    If you have a step that requires a GPU, there are two things you need to do:

    1. First, you'll need to ensure that the :class:`BeakerExecutor` can install your dependencies the right way
    to support the GPU hardware. There are usually two ways to do this: use a Docker image that comes
    with a proper installation of your hardware-specific dependencies (e.g. PyTorch), or add a conda
    ``environment.yml`` file to your project that specifies the proper version of those dependencies.

    If you go with first option you don't necessarily need to build your own Docker image.
    If PyTorch is the only hardware-specific dependency you have, you could just use
    one of AI2's pre-built PyTorch images. Just add these lines to your ``tango.yml`` file:

    .. code:: diff

         executor:
           type: beaker
           beaker_workspace: ai2/my-workspace
        +  docker_image: ghcr.io/allenai/pytorch:1.12.0-cuda11.3-python3.9
        +  venv_name: base
           clusters:
             - ai2/general-cirrascale

    The ``venv_name: base`` line tells the :class:`BeakerExecutor` to use the existing
    conda environment called "base" on the image instead of creating a new one.

    Alternatively, you could use the :data:`default image <DEFAULT_BEAKER_IMAGE>`
    and just add a conda ``environment.yml`` file to the root of your project
    that looks like this:

    .. code:: yaml

        name: torch-env
        channels:
          - pytorch
        dependencies:
          - python=3.9
          - cudatoolkit=11.3
          - numpy
          - pytorch
          - ...

    2. And second, you'll need to specify the GPUs required by each step in the config for that step under
    the :class:`step_resources <tango.step.StepResources>` parameter. For example,

    .. code:: json

        "steps": {
            "train": {
                "type": "torch::train",
                "step_resources": {
                    "gpu_count": 1
                }
            }
        }

    """

    GITHUB_TOKEN_SECRET_NAME: str = "TANGO_GITHUB_TOKEN"

    BEAKER_TOKEN_SECRET_NAME: str = "BEAKER_TOKEN"

    RESULTS_DIR: str = "/tango/output"

    ENTRYPOINT_DIR: str = "/tango/entrypoint"

    ENTRYPOINT_FILENAME: str = "entrypoint.sh"

    INPUT_DIR: str = "/tango/input"

    STEP_GRAPH_FILENAME: str = "config.json"

    DEFAULT_BEAKER_IMAGE: str = "ai2/conda"
    """
    The default image. Used if neither ``beaker_image`` nor ``docker_image`` are set.
    """

    DEFAULT_NFS_DRIVE = "/net/nfs.cirrascale"

    def __init__(
        self,
        workspace: Workspace,
        clusters: List[str],
        include_package: Optional[Sequence[str]] = None,
        beaker_workspace: Optional[str] = None,
        github_token: Optional[str] = None,
        beaker_image: Optional[str] = None,
        docker_image: Optional[str] = None,
        datasets: Optional[List[DataMount]] = None,
        env_vars: Optional[List[EnvVar]] = None,
        venv_name: Optional[str] = None,
        parallelism: Optional[int] = -1,
        install_cmd: Optional[str] = None,
        priority: str = "normal",
        allow_dirty: bool = False,
        **kwargs,
    ):
        # Pre-validate arguments.
        if beaker_image is None and docker_image is None:
            beaker_image = self.DEFAULT_BEAKER_IMAGE
        elif (beaker_image is None) == (docker_image is None):
            raise ConfigurationError(
                "Either 'beaker_image' or 'docker_image' must be specified for BeakerExecutor, but not both."
            )

        from tango.workspaces import LocalWorkspace, MemoryWorkspace

        if isinstance(workspace, MemoryWorkspace):
            raise ConfigurationError(
                "You cannot use the `MemoryWorkspace` with the `BeakerExecutor`! "
                "Please specify a different workspace."
            )
        elif isinstance(workspace, LocalWorkspace):
            if str(workspace.dir).startswith(self.DEFAULT_NFS_DRIVE):
                # Mount the NFS drive if not mount already.
                datasets = datasets or []
                if not datasets or not any(
                    [
                        dm.source.host_path is not None
                        and dm.source.host_path.startswith(self.DEFAULT_NFS_DRIVE)
                        for dm in datasets
                    ]
                ):
                    nfs_mount = DataMount.new(
                        self.DEFAULT_NFS_DRIVE, host_path=self.DEFAULT_NFS_DRIVE
                    )
                    datasets.append(nfs_mount)
            else:
                warnings.warn(
                    "It appears that you're using a `LocalWorkspace` on a directory that is not an NFS drive. "
                    "If the `BeakerExecutor` cannot access this directory from Beaker, your results will be lost.",
                    UserWarning,
                )

        super().__init__(workspace, include_package=include_package, parallelism=parallelism)
        self.beaker = Beaker.from_env(default_workspace=beaker_workspace, session=True, **kwargs)
        self.beaker_image = beaker_image
        self.docker_image = docker_image
        self.datasets = datasets
        self.env_vars = env_vars
        self.clusters = clusters
        self.venv_name = venv_name
        self.install_cmd = install_cmd
        self.priority = priority
        self.allow_dirty = allow_dirty
        self._is_cancelled = threading.Event()
        self._latest_clusters_used: List[str] = []

        try:
            self.github_token: str = github_token or os.environ["GITHUB_TOKEN"]
        except KeyError:
            raise ConfigurationError(
                "A GitHub personal access token with the 'repo' scope is required. "
                "This can be set with the 'github_token' argument to the BeakerExecutor, "
                "or as the environment variable 'GITHUB_TOKEN'."
            )

        # Ensure entrypoint dataset exists.
        self._ensure_entrypoint_dataset()

    def check_repo_state(self):
        if not self.allow_dirty:
            # Make sure repository is clean, if we're in one.
            try:
                # Check for uncommitted changes.
                repo = Repo(".")
                if repo.is_dirty():
                    raise ExecutorError(
                        "You have uncommitted changes! Commit your changes or use the 'allow_dirty' option."
                    )

                # Check for un-pushed commits.
                remote_name = repo.remote().name
                git = Git(".")
                if git.log([f"{remote_name}..HEAD", "--not", "--remotes", "--oneline"]):
                    raise ExecutorError(
                        "You have unpushed changes! Push your changes or use the 'allow_dirty' option."
                    )
            except InvalidGitRepositoryError:
                raise ExecutorError(
                    "It appears you're not in a valid git repository. "
                    "The Beaker executor requires a git repository."
                )
            except GitCommandError:
                pass

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        import concurrent.futures

        self.check_repo_state()

        self._is_cancelled.clear()

        # These will hold the final results which we'll update along the way.
        successful: Dict[str, ExecutionMetadata] = {}
        failed: Dict[str, ExecutionMetadata] = {}
        not_run: Dict[str, ExecutionMetadata] = {}

        # Keeps track of steps that are next up to run on Beaker.
        steps_to_run: Set[str] = set()
        # These are steps that have been submitted to Beaker but haven't completed yet.
        submitted_steps: Set[str] = set()
        # Futures for tracking the Beaker runs for each step.
        step_futures: List[concurrent.futures.Future] = []

        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()

        def update_steps_to_run():
            nonlocal steps_to_run, not_run
            for step_name, step in step_graph.items():
                if (
                    step_name in submitted_steps
                    or step_name in successful
                    or step_name in failed
                    or step_name in not_run
                ):
                    # Make sure step is no longer in queue.
                    steps_to_run.discard(step_name)  # This does NOT raise KeyError if not found
                else:
                    # Check dependencies.
                    for dependency in step.dependencies:
                        if dependency.name not in successful and dependency.cache_results:
                            if dependency.name in failed or dependency.name in not_run:
                                # A dependency failed or can't be run, so this step can't be run.
                                not_run[step_name] = ExecutionMetadata()
                                steps_to_run.discard(step_name)
                            break
                    else:
                        # Dependencies are OK, so we can run this step now.
                        if step.cache_results or step in uncacheable_leaf_steps:
                            steps_to_run.add(step_name)

        def make_future_done_callback(step_name: str):
            def future_done_callback(future: concurrent.futures.Future):
                nonlocal successful, failed
                try:
                    exc = future.exception()
                    if exc is None:
                        successful[step_name] = ExecutionMetadata(
                            result_location=self.workspace.step_info(
                                step_graph[step_name]
                            ).result_location,
                            logs_location=future.result(),
                        )
                    elif isinstance(exc, StepFailedError):
                        failed[step_name] = ExecutionMetadata(logs_location=exc.experiment_url)
                    elif isinstance(exc, ExecutorError):
                        failed[step_name] = ExecutionMetadata()
                    else:
                        log_exception(exc, logger)
                        failed[step_name] = ExecutionMetadata()
                except concurrent.futures.TimeoutError as exc:
                    log_exception(exc, logger)
                    failed[step_name] = ExecutionMetadata()

            return future_done_callback

        update_steps_to_run()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallelism or None
            ) as pool:
                while steps_to_run or step_futures:
                    # Submit steps left to run.
                    for step_name in steps_to_run:
                        future = pool.submit(
                            self._execute_sub_graph_for_step, step_graph, step_name, True
                        )
                        future.add_done_callback(make_future_done_callback(step_name))
                        step_futures.append(future)
                        submitted_steps.add(step_name)

                    # Wait for something to happen.
                    _, not_done = concurrent.futures.wait(
                        step_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # Update the list of running futures.
                    step_futures.clear()
                    step_futures = list(not_done)

                    # Update the step queue.
                    update_steps_to_run()
        except (KeyboardInterrupt, CancellationError):
            if step_futures:
                cli_logger.warning("Received interrupt, canceling steps...")
                self._is_cancelled.set()
                concurrent.futures.wait(step_futures)
            raise
        finally:
            self._is_cancelled.clear()

        # NOTE: The 'done callback' added to each future is executed in a thread,
        # and so might not complete before the last 'update_steps_to_run()' is called
        # in the loop above. Therefore we have to call 'update_steps_to_run()'
        # one last time here to ensure the 'not_run' set is up-to-date.
        update_steps_to_run()

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)

    def execute_sub_graph_for_step(
        self, step_graph: StepGraph, step_name: str, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        self.check_repo_state()
        try:
            experiment_url = self._execute_sub_graph_for_step(step_graph, step_name)
        except Exception as exc:
            if isinstance(exc, StepFailedError):
                return ExecutorOutput(
                    failed={step_name: ExecutionMetadata(logs_location=exc.experiment_url)}
                )
            elif isinstance(exc, ExecutorError):
                return ExecutorOutput(failed={step_name: ExecutionMetadata()})
            else:
                log_exception(exc, logger)
                return ExecutorOutput(failed={step_name: ExecutionMetadata()})
        else:
            return ExecutorOutput(
                successful={
                    step_name: ExecutionMetadata(
                        result_location=self.workspace.step_info(
                            step_graph[step_name]
                        ).result_location,
                        logs_location=experiment_url,
                    )
                }
            )

    def _check_if_cancelled(self):
        if self._is_cancelled.is_set():
            raise RunCancelled

    def _execute_sub_graph_for_step(
        self,
        step_graph: StepGraph,
        step_name: str,
        in_thread: bool = False,
    ) -> Optional[str]:
        if not in_thread:
            self._is_cancelled.clear()

        step = step_graph[step_name]

        if step.cache_results and step in self.workspace.step_cache:
            cli_logger.info(
                '[green]\N{check mark} Found output for step [bold]"%s"[/] in cache...[/]',
                step_name,
            )
            return None

        step.log_starting()

        # Initialize experiment and task spec.
        spec = self._build_experiment_spec(step_graph, step_name)
        self._check_if_cancelled()

        # Create experiment.
        experiment_name = (
            f"{Constants.STEP_EXPERIMENT_PREFIX}{step.unique_id}-{str(uuid.uuid4())[:8]}"
        )
        experiment = self.beaker.experiment.create(experiment_name, spec)
        experiment_url = self.beaker.experiment.url(experiment)
        cli_logger.info(
            '[blue]\N{black rightwards arrow} Submitted Beaker experiment [b]%s[/] for step [b]"%s"[/]...[/]',
            experiment_url,
            step_name,
        )

        # Follow the experiment and stream the logs until it completes.
        try:
            self.beaker.experiment.wait_for(experiment, strict=True, quiet=True, poll_interval=2.0)
        except (JobFailedError, TaskStoppedError):
            cli_logger.error(
                '[red]\N{ballot x} Step [b]"%s"[/] failed. You can check the logs at [b]%s[/][/]',
                step_name,
                experiment_url,
            )
            raise StepFailedError(
                f'Beaker job for step "{step_name}" failed. '
                f"You can check the logs at {experiment_url}",
                experiment_url,
            )
        except (KeyboardInterrupt, CancellationError):
            cli_logger.warning(
                'Stopping Beaker experiment [cyan]%s[/] for step [b]"%s"[/] (%s)',
                experiment_url,
                step_name,
                step.unique_id,
            )
            self.beaker.experiment.stop(experiment)
            raise
        else:
            step.log_finished()
            return experiment_url

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

        entrypoint_dataset_name = (
            f"{Constants.ENTRYPOINT_DATASET_PREFIX}{workspace_id}-{sha256_hash.hexdigest()[:6]}"
        )
        tmp_entrypoint_dataset_name = (
            f"{Constants.ENTRYPOINT_DATASET_PREFIX}{str(uuid.uuid4())}-tmp"
        )

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
                    tmp_entrypoint_dataset = self.beaker.dataset.create(
                        tmp_entrypoint_dataset_name, entrypoint_path, quiet=True
                    )
                    entrypoint_dataset = self.beaker.dataset.rename(
                        tmp_entrypoint_dataset, entrypoint_dataset_name
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
        step_graph_dataset_name = f"{Constants.STEP_GRAPH_DATASET_PREFIX}{str(uuid.uuid4())}"
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname)
                path = tmpdir / self.STEP_GRAPH_FILENAME
                step_graph.to_file(path, include_unique_id=True)
                dataset = self.beaker.dataset.create(step_graph_dataset_name, path, quiet=True)
        except DatasetConflict:  # could be in a race with another `tango` process.
            time.sleep(1.0)
            dataset = self.beaker.dataset.get(step_graph_dataset_name)
        return dataset

    def _ensure_cluster(self, task_resources: TaskResources) -> str:
        cluster_to_use: Optional[str] = None
        if not self.clusters:
            raise ConfigurationError("At least one cluster is required in 'clusters'")
        elif len(self.clusters) == 1:
            cluster_to_use = self.clusters[0]
        else:

            def recency_ranking(cluster_name: str):
                try:
                    return self._latest_clusters_used.index(cluster_name)
                except ValueError:
                    return -1

            available_clusters = sorted(
                self.beaker.cluster.filter_available(task_resources, *self.clusters),
                key=lambda x: (x.queued_jobs, recency_ranking(x.cluster.full_name)),
            )

            if available_clusters:
                cluster_to_use = available_clusters[0].cluster.full_name
                # Move cluster to the end of `self._latest_clusters_used`
                try:
                    self._latest_clusters_used.remove(cluster_to_use)  # type: ignore
                except ValueError:
                    pass
                self._latest_clusters_used.append(cluster_to_use)  # type: ignore
                logger.debug(f"Using cluster '{cluster_to_use}'")
            else:
                cluster_to_use = self.clusters[0]
                logger.debug(
                    "No clusters currently have enough free resources available. "
                    "Will use '%' anyway.",
                    cluster_to_use,
                )

        if cluster_to_use is None:
            cluster_to_use = self.clusters[0]

        return cluster_to_use

    def _build_experiment_spec(self, step_graph: StepGraph, step_name: str) -> ExperimentSpec:
        from tango.common.logging import TANGO_LOG_LEVEL

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
        git = GitMetadata.check_for_repo()
        if (
            git is None
            or git.commit is None
            or git.remote is None
            or "github.com" not in git.remote
        ):
            raise ExecutorError(
                f"Missing git data for step '{step.unique_id}'. "
                f"BeakerExecutor requires a git repository with a GitHub remote."
            )
        try:
            github_account, github_repo = self._parse_git_remote(git.remote)
        except ValueError:
            raise ExecutorError("BeakerExecutor requires a git repository with a GitHub remote.")
        git_ref = git.commit
        cli_logger.info(
            "[blue]\N{black rightwards arrow} Using source code from "
            '[b]https://github.com/%s/%s/commit/%s[/] for step [b]"%s"[/][/]',
            github_account,
            github_repo,
            git_ref,
            step_name,
        )
        self._check_if_cancelled()

        # Ensure dataset with the entrypoint script exists and get it.
        entrypoint_dataset = self._ensure_entrypoint_dataset()
        self._check_if_cancelled()

        # Create dataset for step graph.
        step_graph_dataset = self._ensure_step_graph_dataset(sub_graph)
        self._check_if_cancelled()

        # Write the GitHub token secret.
        self.beaker.secret.write(self.GITHUB_TOKEN_SECRET_NAME, self.github_token)
        self._check_if_cancelled()

        # Write the Beaker token secret.
        self.beaker.secret.write(self.BEAKER_TOKEN_SECRET_NAME, self.beaker.config.user_token)
        self._check_if_cancelled()

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
                command += ["-i", package, "--log-level", TANGO_LOG_LEVEL or "debug"]

        # Get cluster to use.
        cluster = self._ensure_cluster(task_resources)
        self._check_if_cancelled()

        # Ignore the patch version.
        # E.g. '3.9.7' -> '3.9'
        python_version = step_info.environment.python
        python_version = python_version[: python_version.find(".", python_version.find(".") + 1)]

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
                priority=self.priority,
            )
            .with_env_var(name="TANGO_VERSION", value=VERSION)
            .with_env_var(name="GITHUB_TOKEN", secret=self.GITHUB_TOKEN_SECRET_NAME)
            .with_env_var(name="BEAKER_TOKEN", secret=self.BEAKER_TOKEN_SECRET_NAME)
            .with_env_var(name="GITHUB_REPO", value=f"{github_account}/{github_repo}")
            .with_env_var(name="GIT_REF", value=git_ref)
            .with_env_var(name="PYTHON_VERSION", value=python_version)
            .with_dataset(self.ENTRYPOINT_DIR, beaker=entrypoint_dataset.id)
            .with_dataset(self.INPUT_DIR, beaker=step_graph_dataset.id)
        )

        if self.venv_name is not None:
            task_spec = task_spec.with_env_var(name="VENV_NAME", value=self.venv_name)

        if self.install_cmd is not None:
            task_spec = task_spec.with_env_var(name="INSTALL_CMD", value=self.install_cmd)

        return ExperimentSpec(tasks=[task_spec])
