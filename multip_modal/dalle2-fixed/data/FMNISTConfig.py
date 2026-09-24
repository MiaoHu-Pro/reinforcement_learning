"""Backward-compatible imports for the old configuration module.

New code should import from the root-level :mod:`config` module. Keeping this
shim means existing notebooks and scripts continue to work unchanged.
"""

from config import (  # noqa: F401
    CLIPConfig,
    DALLE2_DIR,
    DecoderConfig,
    Dalle2Config,
    FMNISTConfig,
    PROJECT_ROOT,
    PriorConfig,
    add_config_arguments,
    add_dataset_arguments,
    automatic_run_name,
    config_from_args,
    configure_dataset,
    effective_config_dict,
    finalize_experiment,
    load_effective_config,
    mark_training_stage_complete,
    prepare_training_stage,
    save_effective_config,
    training_stage_is_complete,
    validate_config,
)
