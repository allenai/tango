from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.ops.lamb import FusedLamb

from tango.integrations.torch import Optimizer

Optimizer.register("deepspeed::Adagrad")(DeepSpeedCPUAdagrad)
Optimizer.register("deepspeed::Adam")(DeepSpeedCPUAdam)
Optimizer.register("deepspeed::FusedAdam")(FusedAdam)
Optimizer.register("deepspeed::FusedLamb")(FusedLamb)
