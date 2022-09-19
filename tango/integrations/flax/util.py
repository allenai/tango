from typing import Any, Union

import jax


def get_PRNGkey(seed: int = 42) -> Union[Any, jax.random.PRNGKeyArray]:
    """
    Utility function to create a pseudo-random number generator key
    given a seed.
    """
    return jax.random.PRNGKey(seed)


def get_multiple_keys(key, multiple: int = 1) -> Union[Any, jax.random.PRNGKeyArray]:
    """
    Utility function to split a PRNG key into multiple new keys.
    Used in distributed training.
    """
    return jax.random.split(key, multiple)
