import jax


def GetPRNGkey(seed: int = 42):
    return jax.random.PRNGKey(seed)


# Utility function to generate multiple keys
def GetMultipleKeys(key, multiple: int = 1):
    key, *subkeys = jax.random.split(key, multiple)
    return subkeys
