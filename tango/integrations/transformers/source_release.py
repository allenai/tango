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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, result)
        return result
