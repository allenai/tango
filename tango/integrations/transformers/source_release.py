import shutil
import tarfile
from pathlib import Path

from cached_path import cached_path, check_tarfile

from tango import Step


@Step.register("transformers_source_release")
class TransformersSourceRelease(Step[Path]):
    VERSION = "001"
    CACHEABLE = True

    def run(self, hf_version: str = "4.9.2") -> Path:  # type: ignore
        url = f"https://github.com/huggingface/transformers/archive/refs/tags/v{hf_version}.tar.gz"
        result = self.work_dir / "transformers"
        if result.exists():
            shutil.rmtree(result)
        with tarfile.open(cached_path(url)) as tar_file:
            check_tarfile(tar_file)
            tar_file.extractall(result)
        return result
