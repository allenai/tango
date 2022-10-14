import os

from tango.common.testing.steps import FloatStep
from tango.integrations.gcs.step_cache import GCSStepCache

# TODO: for now
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/Users/akshitab/local/keys/allennlp-cloud-storage-key.json"


def test_step_cache(bucket_name: str = "allennlp-gcs-bucket-1"):
    cache = GCSStepCache(bucket_name=bucket_name)
    step = FloatStep(result=1.0)
    cache[step] = 1.0
    assert step in cache
    assert len(cache) == 1
    assert FloatStep(result=2.0) not in cache
    assert cache[step] == 1.0


if __name__ == "__main__":
    test_step_cache("allennlp-gcs-bucket-1")
