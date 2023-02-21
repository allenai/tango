"""
Classes and utility functions for GSWorkspace and GSStepCache.
"""
import atexit
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import google.auth
from google.api_core import exceptions
from google.cloud import storage
from google.oauth2.credentials import Credentials

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError
from tango.common.remote_utils import RemoteConstants
from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


def empty_bucket(bucket_name: str):
    """
    Removes all the tango-related blobs from the specified bucket.
    Used for testing.
    """
    credentials, project = google.auth.default()
    client = storage.Client(project=project, credentials=credentials)
    bucket = client.bucket(bucket_name)
    try:
        bucket.delete_blobs(list(bucket.list_blobs(prefix="tango-")))
    except exceptions.NotFound:
        pass


def empty_datastore(namespace: str):
    """
    Removes all the tango-related entities from the specified namespace in datastore.
    Used for testing.
    """
    from google.cloud import datastore

    credentials, project = google.auth.default()
    client = datastore.Client(project=project, credentials=credentials, namespace=namespace)
    run_query = client.query(kind="run")
    run_query.keys_only()
    keys = [entity.key for entity in run_query.fetch()]
    stepinfo_query = client.query(kind="stepinfo")
    stepinfo_query.keys_only()
    keys += [entity.key for entity in stepinfo_query.fetch()]
    client.delete_multi(keys)


@dataclass
class GSArtifact:
    """
    A GSArtifact object is used for representing storage objects in google cloud storage.
    """

    name: str
    """
    Name of the artifact.
    """
    artifact_path: str
    """
    Remote location url for the artifact.
    """
    created: datetime.datetime
    """
    Time of creation.
    """
    committed: bool
    """
    If set to True, no further changes to the remote artifact are allowed.
    If set to False, it means that the artifact is under construction.
    """


class GSArtifactConflict(TangoError):
    """
    Error denoting that the storage artifact already exists.
    """

    pass


class GSArtifactNotFound(TangoError):
    """
    Error denoting that the storage artifact does not exist.
    """

    pass


class GSArtifactWriteError(TangoError):
    """
    Error denoting that there was an issue writing the artifact to the remote storage.
    """

    pass


class GSClient:
    """
    A client for interacting with Google Cloud Storage. The authorization works by
    providing OAuth2 credentials.

    :param bucket_name: The name of the Google Cloud bucket to use.
    :param credentials: OAuth2 credentials can be provided. If not provided, default
        gcloud credentials are inferred.
    :param project: Optionally, the project ID can be provided. This is not essential
        for `google.cloud.storage` API, since buckets are at the account level, rather
        than the project level.
    """

    placeholder_file = ".placeholder"
    """
    The placeholder file is used for creation of a folder in the cloud bucket,
    as empty folders are not allowed. It is also used as a marker for the creation
    time of the folder, hence we use a separate file to mark the artifact as
    uncommitted.
    """

    uncommitted_file = ".uncommitted"
    """
    The uncommitted file is used to denote an artifact under construction.
    """

    settings_file = "settings.json"
    """
    This file is for storing metadata like version information, etc.
    """

    NUM_CONCURRENT_WORKERS: int = 9

    def __init__(
        self,
        bucket_name: str,
        credentials: Optional[Credentials] = None,
        project: Optional[str] = None,
    ):
        if not credentials:
            credentials, project = google.auth.default()

        self.storage = storage.Client(project=project, credentials=credentials)
        self.bucket_name = bucket_name

        blob = self.storage.bucket(bucket_name).blob(self.settings_file)  # no HTTP request yet
        try:
            with blob.open("r") as file_ref:
                json.load(file_ref)
        except exceptions.NotFound:
            settings = {"version": 1}
            with blob.open("w") as file_ref:
                json.dump(settings, file_ref)

    def url(self, artifact: Optional[str] = None):
        """
        Returns the remote url of the storage artifact.
        """
        path = f"gs://{self.bucket_name}"
        if artifact is not None:
            path = f"{path}/{artifact}"
        return path

    @classmethod
    def _convert_blobs_to_artifact(cls, blobs: List[storage.Blob]) -> GSArtifact:
        """
        Converts a list of `google.cloud.storage.Blob` to a `GSArtifact`.
        """
        name: str
        artifact_path: str
        created: datetime.datetime
        committed: bool = True

        for blob in blobs:
            if blob.name.endswith(cls.placeholder_file):
                created = blob.time_created
                name = blob.name.replace("/" + cls.placeholder_file, "")
                artifact_path = name  # does not contain bucket info here.
            elif blob.name.endswith(cls.uncommitted_file):
                committed = False

        assert name is not None, "Folder is not a GSArtifact, should not have happened."
        return GSArtifact(name, artifact_path, created, committed)

    @classmethod
    def from_env(cls, bucket_name: str):
        """
        Constructs the client object from the environment, using default credentials.
        """
        return cls(bucket_name)

    def get(self, artifact: Union[str, GSArtifact]) -> GSArtifact:
        """
        Returns a `GSArtifact` object created by fetching the artifact's information
        from remote location.
        """
        if isinstance(artifact, str):
            path = artifact
        else:
            # We have an artifact, and we recreate it with refreshed info.
            path = artifact.artifact_path

        blobs = list(self.storage.bucket(self.bucket_name).list_blobs(prefix=path))
        if len(blobs) > 0:
            return self._convert_blobs_to_artifact(blobs)
        else:
            raise GSArtifactNotFound()

    @classmethod
    def _gs_path(cls, *args):
        """
        Returns path within google cloud storage bucket. We use this since we cannot
        use `os.path.join` for cloud storage paths.
        """
        return "/".join(args)

    def create(self, artifact: str):
        """
        Creates a new artifact in the remote location. By default, it is uncommitted.
        """
        bucket = self.storage.bucket(self.bucket_name)
        # gives refreshed information
        if bucket.blob(self._gs_path(artifact, self.placeholder_file)).exists():
            raise GSArtifactConflict(f"{artifact} already exists!")
        else:
            # Additional safety check
            if bucket.blob(self._gs_path(artifact, self.uncommitted_file)).exists():
                raise GSArtifactConflict(f"{artifact} already exists!")
            bucket.blob(self._gs_path(artifact, self.placeholder_file)).upload_from_string("")
            bucket.blob(self._gs_path(artifact, self.uncommitted_file)).upload_from_string("")
        return self._convert_blobs_to_artifact(list(bucket.list_blobs(prefix=artifact)))

    def delete(self, artifact: GSArtifact):
        """
        Removes the artifact from the remote location.
        """
        bucket = self.storage.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs(prefix=artifact.artifact_path))
        bucket.delete_blobs(blobs)

    def upload(self, artifact: Union[str, GSArtifact], objects_dir: Path):
        """
        Writes the contents of objects_dir to the remote artifact location.
        """
        if isinstance(artifact, str):
            folder_path = artifact
        else:
            folder_path = artifact.artifact_path

        source_path = str(objects_dir)

        def _sync_blob(source_file_path: str, target_file_path: str):
            blob = self.storage.bucket(self.bucket_name).blob(target_file_path)
            blob.upload_from_filename(source_file_path)

        import concurrent.futures

        try:
            # TODO: google-cloud-storage==2.7.0 has added a preview feature called `transfer_manager`
            # which allows for concurrent uploads and downloads. We should upgrade to this once
            # it is more robustly supported. Also update in `download()`.
            if os.path.isdir(source_path):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.NUM_CONCURRENT_WORKERS, thread_name_prefix="GSClient.upload()-"
                ) as executor:
                    upload_futures = []
                    for dirpath, _, filenames in os.walk(source_path):
                        for filename in filenames:
                            source_file_path = os.path.join(dirpath, filename)
                            target_file_path = self._gs_path(
                                folder_path, source_file_path.replace(source_path + "/", "")
                            )
                        upload_futures.append(
                            executor.submit(_sync_blob, source_file_path, target_file_path)
                        )
                    for future in concurrent.futures.as_completed(upload_futures):
                        future.result()
            else:
                source_file_path = source_path
                target_file_path = self._gs_path(folder_path, os.path.basename(source_file_path))
                _sync_blob(source_file_path, target_file_path)
        except Exception:
            raise GSArtifactWriteError()

    def commit(self, artifact: Union[str, GSArtifact]):
        """
        Marks the artifact as committed. No further changes to the artifact are allowed.
        """
        if isinstance(artifact, str):
            folder_path = artifact
        else:
            folder_path = artifact.artifact_path
        bucket = self.storage.bucket(self.bucket_name)
        try:
            bucket.delete_blob(self._gs_path(folder_path, self.uncommitted_file))
        except exceptions.NotFound:
            if not bucket.blob(self._gs_path(folder_path, self.placeholder_file)).exists():
                raise GSArtifactNotFound()
            # Otherwise, already committed. No change.

    def download(self, artifact: GSArtifact, target_dir: PathOrStr):
        """
        Writes the contents of the remote artifact to the `target_dir`.
        """
        assert (
            self.storage.bucket(self.bucket_name)
            .blob(self._gs_path(artifact.artifact_path, self.placeholder_file))
            .exists()
        )

        def _fetch_blob(blob: storage.Blob):
            source_path = blob.name.replace(artifact.artifact_path + "/", "")
            target_path = os.path.join(target_dir, source_path)
            if not os.path.exists(os.path.dirname(target_path)):
                os.mkdir(os.path.dirname(target_path))
            blob.download_to_filename(target_path)

        import concurrent.futures

        bucket = self.storage.bucket(self.bucket_name)
        bucket.update()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.NUM_CONCURRENT_WORKERS, thread_name_prefix="GSClient.download()-"
            ) as executor:
                download_futures = []
                for blob in bucket.list_blobs(self.bucket_name, prefix=artifact.artifact_path):
                    download_futures.append(executor.submit(_fetch_blob, blob))
                for future in concurrent.futures.as_completed(download_futures):
                    future.result()
        except exceptions.NotFound:
            raise GSArtifactWriteError()

    def artifacts(self, prefix: str, uncommitted: bool = True) -> List[GSArtifact]:
        """
        Lists all the artifacts within the remote storage, based on
        `match` and `uncommitted` criteria. These can include steps and runs.
        """
        list_of_artifacts = []
        for folder_name in self.storage.list_blobs(
            self.bucket_name, prefix=prefix, delimiter="/"
        )._get_next_page_response()["prefixes"]:
            artifact = self._convert_blobs_to_artifact(
                list(self.storage.list_blobs(self.bucket_name, prefix=folder_name))
            )
            if not uncommitted:
                if not artifact.committed:
                    continue
            list_of_artifacts.append(artifact)
        return list_of_artifacts


def get_credentials(credentials: Optional[Union[str, Credentials]] = None) -> Credentials:
    """
    :param credentials:
        * if OAuth2 credentials are provided, they are returned.
        * if `str`, it can be either a file path or a json string of credentials dict.
        * if `None`, credentials are inferred from the environment.

    More details on Google Cloud credentials can be found here:
    https://googleapis.dev/python/google-auth/latest/user-guide.html#service-account-private-key-files,
    and https://googleapis.dev/python/google-api-core/latest/auth.html
    """

    # BeakerExecutor uses GOOGLE_TOKEN
    credentials = os.environ.get("GOOGLE_TOKEN", credentials)
    if credentials is not None:
        # Path to the credentials file has been provided
        if isinstance(credentials, str) and credentials.endswith(".json"):
            with open(credentials) as file_ref:
                credentials = file_ref.read()
        try:
            # If credentials dict has been passed as a json string
            credentials_dict = json.loads(credentials)
            credentials_dict.pop("type", None)

            # sometimes the credentials dict may not contain `token` and `token_uri` keys,
            # but `Credentials()` needs the parameter.
            token = credentials_dict.pop("token", None)
            token_uri = credentials_dict.pop("token_uri", "https://oauth2.googleapis.com/token")
            credentials = Credentials(token=token, token_uri=token_uri, **credentials_dict)
        except json.decoder.JSONDecodeError:
            # It is not a json string.
            # We use this string because BeakerExecutor cannot write a None secret.
            if credentials == "default":
                credentials = None
    if not credentials:
        # Infer default credentials
        credentials, _ = google.auth.default()
    return credentials


def get_client(
    bucket_name: str,
    credentials: Optional[Union[str, Credentials]] = None,
    project: Optional[str] = None,
) -> GSClient:
    """
    Returns a `GSClient` object for a google cloud bucket.
    """
    credentials = get_credentials(credentials)
    return GSClient(bucket_name, credentials=credentials, project=project)


class Constants(RemoteConstants):
    pass


class GCSStepLock:
    """
    Google Cloud offers consistency https://cloud.google.com/storage/docs/consistency,
    so we can use lock files.
    """

    def __init__(
        self,
        client: GSClient,
        step: Union[str, StepInfo, Step],
    ):
        self._client = client
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._lock_artifact_name = RemoteConstants.step_lock_artifact_name(step)
        self._lock_artifact: Optional[GSArtifact] = None
        self.lock_artifact_url = self._client.url(self._lock_artifact_name)

    def acquire(self, timeout=None, poll_interval: float = 2.0, log_interval: float = 30.0) -> None:
        if self._lock_artifact is not None:
            return
        start = time.monotonic()
        last_logged = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_artifact = self._client.create(self._lock_artifact_name)
                atexit.register(self.release)

            except GSArtifactConflict:
                if last_logged is None or last_logged - start >= log_interval:
                    logger.warning(
                        "Waiting to acquire lock artifact for step '%s':\n\n%s\n\n"
                        "This probably means the step is being run elsewhere, but if you're sure it isn't "
                        "you can just delete the lock artifact, using the command: \n`gsutil rm -r %s`",
                        self._step_id,
                        self.lock_artifact_url,
                        self.lock_artifact_url,
                    )
                    last_logged = time.monotonic()
                time.sleep(poll_interval)
                continue
            else:
                break
        else:
            raise TimeoutError(
                f"Timeout error occurred while waiting to acquire artifact lock for step '{self._step_id}':\n\n"
                f"{self.lock_artifact_url}\n\n"
                f"This probably means the step is being run elsewhere, but if you're sure it isn't you can "
                f"just delete the lock, using the command: \n`gsutil rm -r {self.lock_artifact_url}`"
            )

    def release(self):
        if self._lock_artifact is not None:
            try:
                self._client.delete(self._lock_artifact)
            except GSArtifactNotFound:
                # Artifact must have been manually deleted.
                pass
            self._lock_artifact = None
            atexit.unregister(self.release)

    def __del__(self):
        self.release()
