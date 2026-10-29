"""Shipped remote worker recipes independent of installed local environments.

These descriptors identify the controlled portable bundle. Their relative
locations are interpreted only by the remote executor; Runtime.command rejects
SkyPilot descriptors for local execution.
"""

from vla_platform.lifecycle.native_profiles import NATIVE_PROFILES
from vla_platform.lifecycle.runtime import Runtime

PORTABLE_RECIPE_IDS = frozenset({"portable-policy"})


def portable_runtimes():
    return [
        Runtime(
            id="portable-policy",
            label="Portable policy worker",
            execution="skypilot",
            # Internal legacy contract metadata; the accepted target owns actual infrastructure.
            provider="gcp",
            device="cuda",
            gpu_count=1,
            worker_root="workers/vla_cpp",
            training_python="python",
            training_root="workers/smolvla_qlora",
            training_model_ids=["smolvla", *NATIVE_PROFILES, "psi0"],
            vendor="portable-bundle",
            build="portable-bundle",
        )
    ]


def recipe_catalog(catalog):
    # Built-in IDs cannot be redirected through a user-installed runtime definition.
    return catalog.model_copy(
        update={
            "runtimes": [
                *[runtime for runtime in catalog.runtimes if runtime.id not in PORTABLE_RECIPE_IDS],
                *portable_runtimes(),
            ]
        }
    )
