from typing import Any, Union

import jax


def get_PRNGkey(seed: int = 42) -> Union[Any, jax.random.PRNGKeyArray]:
    return jax.random.PRNGKey(seed)


def get_multiple_keys(key, multiple: int = 1) -> Union[Any, jax.random.PRNGKeyArray]:
    return jax.random.split(key, multiple)
