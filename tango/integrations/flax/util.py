import jax
from typing import List


def GetPRNGkey(seed: int = 42) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(seed)


# Utility function to generate multiple keys
def GetMultipleKeys(key: jax.random.PRNGKey, multiple: int = 1) -> List[jax.random.PRNGKey]:
    key, *subkeys = jax.random.split(key, multiple)
    return subkeys
