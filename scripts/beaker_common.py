import os

from beaker import Beaker

BEAKER_WORKSPACE = os.environ.get("BEAKER_WORKSPACE", "ai2/tango-testing")
BEAKER_CLOUD_CLUSTER = os.environ.get("BEAKER_CLOUD_CLUSTER", "ai2/tango-gpu-tests")
BEAKER_ON_PREM_CLUSTERS = (
    "ai2/general-cirrascale",
    "ai2/allennlp-cirrascale",
    "ai2/aristo-cirrascale",
    "ai2/mosaic-cirrascale",
    "ai2/s2-cirrascale",
)
BEAKER_IMAGE = os.environ.get("BEAKER_IMAGE", "petew/tango-testing")

beaker = Beaker.from_env(default_workspace=BEAKER_WORKSPACE)
