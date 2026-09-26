import math
import os
from typing import Any, Dict, Mapping, Optional

import h5py
import numpy as np
import torch


FORMAT_NAME = "DeepH-checkpoint"
FORMAT_VERSION = 1

SUPPORTED_OPTIMIZERS = {
    "SGD",
    "Adam",
    "AdamW",
    "Adagrad",
    "RMSprop",
}

OPTIMIZER_STATE_KEYS = {
    "SGD": {
        "momentum_buffer",
    },
    "Adam": {
        "step",
        "exp_avg",
        "exp_avg_sq",
        "max_exp_avg_sq",
    },
    "AdamW": {
        "step",
        "exp_avg",
        "exp_avg_sq",
        "max_exp_avg_sq",
    },
    "Adagrad": {
        "step",
        "sum",
    },
    "RMSprop": {
        "step",
        "square_avg",
        "momentum_buffer",
        "grad_avg",
    },
}

ALLOWED_ROOT_GROUPS = {
    "training",
    "dataset",
    "model",
    "optimizer",
}


class CheckpointError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def _decode_string(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")

    raise CheckpointError(
        f"Expected an HDF5 string, got {type(value)!r}"
    )


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _numpy_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype == torch.bfloat16:
        raise CheckpointError(
            "bfloat16 checkpoint tensors are not supported "
            "by this HDF5 format yet"
        )

    return tensor.numpy()


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all().item())

    return True


def _write_tensor(
    group: h5py.Group,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if not torch.is_tensor(tensor):
        raise CheckpointError(
            f"{name!r} is not a tensor"
        )

    if not _tensor_is_finite(tensor):
        raise CheckpointError(
            f"Refusing to save non-finite tensor {name!r}"
        )

    array = _numpy_from_tensor(tensor)

    dataset = group.create_dataset(
        name,
        data=array,
    )

    dataset.attrs["torch_dtype"] = _torch_dtype_name(
        tensor.dtype
    )


def _read_tensor(
    dataset: h5py.Dataset,
    *,
    expected_shape: Optional[torch.Size] = None,
    expected_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not isinstance(dataset, h5py.Dataset):
        raise CheckpointError(
            f"Expected HDF5 dataset, got {type(dataset)!r}"
        )

    if "torch_dtype" not in dataset.attrs:
        raise CheckpointError(
            f"Missing torch_dtype attribute at {dataset.name}"
        )

    stored_dtype = _decode_string(
        dataset.attrs["torch_dtype"]
    )

    array = np.asarray(dataset[()])

    try:
        tensor = torch.from_numpy(
            np.array(array, copy=True)
        )
    except Exception as exc:
        raise CheckpointError(
            f"Unable to convert {dataset.name} to tensor"
        ) from exc

    if _torch_dtype_name(tensor.dtype) != stored_dtype:
        raise CheckpointError(
            f"Dtype metadata mismatch at {dataset.name}: "
            f"metadata={stored_dtype}, actual={tensor.dtype}"
        )

    if expected_shape is not None:
        _require(
            tuple(tensor.shape) == tuple(expected_shape),
            f"Shape mismatch at {dataset.name}: "
            f"checkpoint={tuple(tensor.shape)}, "
            f"expected={tuple(expected_shape)}",
        )

    if expected_dtype is not None:
        _require(
            tensor.dtype == expected_dtype,
            f"Dtype mismatch at {dataset.name}: "
            f"checkpoint={tensor.dtype}, "
            f"expected={expected_dtype}",
        )

    if not _tensor_is_finite(tensor):
        raise CheckpointError(
            f"Non-finite tensor found at {dataset.name}"
        )

    return tensor


def _optimizer_type(
    optimizer: torch.optim.Optimizer,
) -> str:
    name = type(optimizer).__name__

    if name not in SUPPORTED_OPTIMIZERS:
        raise CheckpointError(
            f"Unsupported optimizer for HDF5 checkpoint: {name}"
        )

    return name


def _parameter_name_maps(
    model: torch.nn.Module,
):
    name_to_parameter = dict(
        model.named_parameters()
    )

    parameter_to_name = {
        parameter: name
        for name, parameter in name_to_parameter.items()
    }

    _require(
        len(name_to_parameter) == len(parameter_to_name),
        "Model contains duplicated parameter objects under "
        "different names; this checkpoint format does not "
        "support that layout",
    )

    return name_to_parameter, parameter_to_name


def _write_scalar_value(
    group: h5py.Group,
    name: str,
    value: Any,
) -> None:
    item = group.create_group(name)

    if value is None:
        item.attrs["type"] = "none"
        return

    if isinstance(value, bool):
        item.attrs["type"] = "bool"
        item.attrs["value"] = bool(value)
        return

    if isinstance(value, int) and not isinstance(value, bool):
        item.attrs["type"] = "int"
        item.attrs["value"] = int(value)
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(
                f"Non-finite float in optimizer setting {name!r}"
            )

        item.attrs["type"] = "float"
        item.attrs["value"] = float(value)
        return

    if isinstance(value, tuple):
        if not all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            for x in value
        ):
            raise CheckpointError(
                f"Unsupported tuple value for {name!r}: {value!r}"
            )

        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if not np.isfinite(array).all():
            raise CheckpointError(
                f"Non-finite tuple in optimizer setting {name!r}"
            )

        item.attrs["type"] = "float_tuple"
        item.create_dataset(
            "value",
            data=array,
        )
        return

    raise CheckpointError(
        f"Unsupported optimizer setting type for {name!r}: "
        f"{type(value)!r}"
    )


def _read_scalar_value(
    group: h5py.Group,
) -> Any:
    _require(
        isinstance(group, h5py.Group),
        "Optimizer setting is not an HDF5 group",
    )

    _require(
        "type" in group.attrs,
        f"Missing optimizer setting type at {group.name}",
    )

    value_type = _decode_string(
        group.attrs["type"]
    )

    if value_type == "none":
        _require(
            len(group) == 0,
            f"Unexpected data under {group.name}",
        )
        return None

    if value_type == "bool":
        _require(
            "value" in group.attrs,
            f"Missing bool value at {group.name}",
        )
        return bool(group.attrs["value"])

    if value_type == "int":
        _require(
            "value" in group.attrs,
            f"Missing int value at {group.name}",
        )
        return int(group.attrs["value"])

    if value_type == "float":
        _require(
            "value" in group.attrs,
            f"Missing float value at {group.name}",
        )

        value = float(group.attrs["value"])

        _require(
            math.isfinite(value),
            f"Non-finite optimizer value at {group.name}",
        )

        return value

    if value_type == "float_tuple":
        _require(
            set(group.keys()) == {"value"},
            f"Malformed tuple setting at {group.name}",
        )

        array = np.asarray(
            group["value"][()],
            dtype=np.float64,
        )

        _require(
            array.ndim == 1,
            f"Tuple setting at {group.name} must be 1-D",
        )

        _require(
            np.isfinite(array).all(),
            f"Non-finite tuple at {group.name}",
        )

        return tuple(
            float(x)
            for x in array.tolist()
        )

    raise CheckpointError(
        f"Unknown optimizer setting type "
        f"{value_type!r} at {group.name}"
    )


def _write_model_state(
    group: h5py.Group,
    model: torch.nn.Module,
) -> None:
    state_dict = model.state_dict()

    group.attrs["count"] = len(state_dict)

    for index, (name, tensor) in enumerate(
        state_dict.items()
    ):
        item = group.create_group(
            f"{index:08d}"
        )

        item.attrs["name"] = name

        _write_tensor(
            item,
            "tensor",
            tensor,
        )


def _read_model_state(
    group: h5py.Group,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    _require(
        isinstance(group, h5py.Group),
        "model/state_dict is not a group",
    )

    current_state = model.state_dict()

    _require(
        "count" in group.attrs,
        "Missing model state count",
    )

    count = int(group.attrs["count"])

    _require(
        count == len(group),
        "Model state entry count does not match HDF5 contents",
    )

    _require(
        count == len(current_state),
        f"Model state count mismatch: "
        f"checkpoint={count}, current={len(current_state)}",
    )

    loaded = {}

    for key in sorted(group.keys()):
        item = group[key]

        _require(
            isinstance(item, h5py.Group),
            f"Malformed model state entry {item.name}",
        )

        _require(
            set(item.keys()) == {"tensor"},
            f"Unexpected objects in {item.name}",
        )

        _require(
            "name" in item.attrs,
            f"Missing model state name in {item.name}",
        )

        name = _decode_string(
            item.attrs["name"]
        )

        _require(
            name not in loaded,
            f"Duplicate model state key {name!r}",
        )

        _require(
            name in current_state,
            f"Unknown model state key {name!r}",
        )

        reference = current_state[name]

        loaded[name] = _read_tensor(
            item["tensor"],
            expected_shape=reference.shape,
            expected_dtype=reference.dtype,
        )

    _require(
        set(loaded.keys()) == set(current_state.keys()),
        "Checkpoint model state keys do not exactly match "
        "the current model",
    )

    return loaded



def _read_pretrained_model_state(
    group: h5py.Group,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Read shape-compatible pretrained tensors with legacy DeepH semantics."""
    _require(isinstance(group, h5py.Group), "model/state_dict is not a group")
    current_state = model.state_dict()
    _require("count" in group.attrs, "Missing model state count")
    count = int(group.attrs["count"])
    _require(count == len(group), "Model state entry count does not match HDF5 contents")

    seen = set()
    loaded = {}
    for key in sorted(group.keys()):
        item = group[key]
        _require(isinstance(item, h5py.Group), f"Malformed model state entry {item.name}")
        _require(set(item.keys()) == {"tensor"}, f"Unexpected objects in {item.name}")
        _require("name" in item.attrs, f"Missing model state name in {item.name}")
        name = _decode_string(item.attrs["name"])
        _require(name not in seen, f"Duplicate model state key {name!r}")
        seen.add(name)
        _require(name in current_state, f"Unknown pretrained model state key {name!r}")

        reference = current_state[name]
        tensor = _read_tensor(item["tensor"])
        if tuple(tensor.shape) == tuple(reference.shape):
            _require(
                tensor.dtype == reference.dtype,
                f"Pretrained model state dtype mismatch for {name!r}: "
                f"checkpoint={tensor.dtype}, current={reference.dtype}",
            )
            loaded[name] = tensor
    return loaded


def _write_optimizer(
    group: h5py.Group,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer_name = _optimizer_type(
        optimizer
    )

    group.attrs["type"] = optimizer_name

    name_to_parameter, parameter_to_name = (
        _parameter_name_maps(model)
    )

    param_groups_group = group.create_group(
        "param_groups"
    )

    param_groups_group.attrs["count"] = len(
        optimizer.param_groups
    )

    for group_index, param_group in enumerate(
        optimizer.param_groups
    ):
        output_group = param_groups_group.create_group(
            f"{group_index:08d}"
        )

        parameters = param_group["params"]

        parameter_names = []

        for parameter in parameters:
            _require(
                parameter in parameter_to_name,
                "Optimizer contains a parameter that does not "
                "belong to the model",
            )

            parameter_names.append(
                parameter_to_name[parameter]
            )

        string_dtype = h5py.string_dtype(
            encoding="utf-8"
        )

        output_group.create_dataset(
            "parameters",
            data=np.asarray(
                parameter_names,
                dtype=object,
            ),
            dtype=string_dtype,
        )

        settings_group = output_group.create_group(
            "settings"
        )

        for key in sorted(param_group.keys()):
            if key == "params":
                continue

            _write_scalar_value(
                settings_group,
                key,
                param_group[key],
            )

    state_group = group.create_group(
        "state"
    )

    allowed_state_keys = OPTIMIZER_STATE_KEYS[
        optimizer_name
    ]

    for parameter, state in optimizer.state.items():
        _require(
            parameter in parameter_to_name,
            "Optimizer state refers to a parameter that "
            "does not belong to the model",
        )

        parameter_name = parameter_to_name[
            parameter
        ]

        unknown_keys = (
            set(state.keys())
            - allowed_state_keys
        )

        _require(
            not unknown_keys,
            f"Unsupported {optimizer_name} state keys for "
            f"{parameter_name!r}: {sorted(unknown_keys)}",
        )

        parameter_group = state_group.create_group(
            parameter_name
        )

        parameter_group.attrs["parameter_shape"] = (
            parameter.shape
        )

        parameter_group.attrs["parameter_dtype"] = (
            _torch_dtype_name(parameter.dtype)
        )

        for state_name, state_value in state.items():
            _require(
                torch.is_tensor(state_value),
                f"Optimizer state "
                f"{parameter_name}/{state_name} is not a tensor",
            )

            _write_tensor(
                parameter_group,
                state_name,
                state_value,
            )


def _restore_optimizer(
    group: h5py.Group,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected_optimizer_name = _optimizer_type(
        optimizer
    )

    _require(
        "type" in group.attrs,
        "Missing optimizer type",
    )

    stored_optimizer_name = _decode_string(
        group.attrs["type"]
    )

    _require(
        stored_optimizer_name == expected_optimizer_name,
        f"Optimizer type mismatch: "
        f"checkpoint={stored_optimizer_name}, "
        f"current={expected_optimizer_name}",
    )

    _require(
        set(group.keys()) == {
            "param_groups",
            "state",
        },
        "Unexpected optimizer HDF5 structure",
    )

    name_to_parameter, _ = _parameter_name_maps(
        model
    )

    param_groups_group = group["param_groups"]

    _require(
        isinstance(param_groups_group, h5py.Group),
        "optimizer/param_groups is not a group",
    )

    _require(
        "count" in param_groups_group.attrs,
        "Missing optimizer param-group count",
    )

    group_count = int(
        param_groups_group.attrs["count"]
    )

    _require(
        group_count == len(optimizer.param_groups),
        f"Optimizer param-group count mismatch: "
        f"checkpoint={group_count}, "
        f"current={len(optimizer.param_groups)}",
    )

    _require(
        group_count == len(param_groups_group),
        "Malformed optimizer param-group table",
    )

    restored_groups = []

    for group_index in range(group_count):
        key = f"{group_index:08d}"

        _require(
            key in param_groups_group,
            f"Missing optimizer param group {key}",
        )

        source_group = param_groups_group[key]

        _require(
            set(source_group.keys()) == {
                "parameters",
                "settings",
            },
            f"Malformed optimizer param group {key}",
        )

        raw_names = source_group[
            "parameters"
        ][()]

        if np.asarray(raw_names).ndim != 1:
            raise CheckpointError(
                f"Parameter-name list in {source_group.name} "
                "must be 1-D"
            )

        parameter_names = [
            _decode_string(value)
            for value in raw_names
        ]

        _require(
            len(parameter_names)
            == len(set(parameter_names)),
            f"Duplicate optimizer parameters in "
            f"{source_group.name}",
        )

        for name in parameter_names:
            _require(
                name in name_to_parameter,
                f"Unknown optimizer parameter {name!r}",
            )

        current_parameter_names = []

        for parameter in optimizer.param_groups[
            group_index
        ]["params"]:
            found_name = None

            for name, candidate in (
                name_to_parameter.items()
            ):
                if candidate is parameter:
                    found_name = name
                    break

            _require(
                found_name is not None,
                "Current optimizer contains a parameter "
                "outside the model",
            )

            current_parameter_names.append(
                found_name
            )

        _require(
            parameter_names
            == current_parameter_names,
            f"Optimizer parameter order mismatch in "
            f"group {group_index}",
        )

        settings_group = source_group[
            "settings"
        ]

        restored_group = {
            "params": list(
                range(len(parameter_names))
            )
        }

        for setting_name in settings_group.keys():
            restored_group[setting_name] = (
                _read_scalar_value(
                    settings_group[
                        setting_name
                    ]
                )
            )

        current_settings = {
            key: value
            for key, value
            in optimizer.param_groups[
                group_index
            ].items()
            if key != "params"
        }

        _require(
            set(restored_group.keys())
            - {"params"}
            == set(current_settings.keys()),
            f"Optimizer setting keys mismatch in "
            f"group {group_index}",
        )

        restored_groups.append(
            restored_group
        )

    state_group = group["state"]

    _require(
        isinstance(state_group, h5py.Group),
        "optimizer/state is not a group",
    )

    allowed_state_keys = OPTIMIZER_STATE_KEYS[
        stored_optimizer_name
    ]

    optimizer.state.clear()

    for parameter_name in state_group.keys():
        _require(
            parameter_name in name_to_parameter,
            f"Optimizer state contains unknown parameter "
            f"{parameter_name!r}",
        )

        parameter = name_to_parameter[
            parameter_name
        ]

        parameter_group = state_group[
            parameter_name
        ]

        _require(
            isinstance(parameter_group, h5py.Group),
            f"Malformed optimizer state for "
            f"{parameter_name!r}",
        )

        unknown_keys = (
            set(parameter_group.keys())
            - allowed_state_keys
        )

        _require(
            not unknown_keys,
            f"Unsupported optimizer state keys for "
            f"{parameter_name!r}: {sorted(unknown_keys)}",
        )

        restored_state = {}

        for state_name in parameter_group.keys():
            tensor = _read_tensor(
                parameter_group[state_name]
            )

            if tensor.ndim != 0:
                _require(
                    tuple(tensor.shape)
                    == tuple(parameter.shape),
                    f"Optimizer tensor shape mismatch for "
                    f"{parameter_name}/{state_name}: "
                    f"{tuple(tensor.shape)} vs "
                    f"{tuple(parameter.shape)}",
                )

                _require(
                    tensor.dtype == parameter.dtype,
                    f"Optimizer tensor dtype mismatch for "
                    f"{parameter_name}/{state_name}: "
                    f"{tensor.dtype} vs {parameter.dtype}",
                )

            restored_state[state_name] = (
                tensor.to(parameter.device)
            )

        optimizer.state[parameter] = (
            restored_state
        )

    for group_index, restored_group in enumerate(
        restored_groups
    ):
        current_params = optimizer.param_groups[
            group_index
        ]["params"]

        new_group = {
            key: value
            for key, value in restored_group.items()
            if key != "params"
        }

        new_group["params"] = current_params

        optimizer.param_groups[
            group_index
        ].clear()

        optimizer.param_groups[
            group_index
        ].update(new_group)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[
        torch.optim.Optimizer
    ],
    *,
    epoch: int,
    best_val_loss: float,
    spinful: bool,
    index_to_Z: torch.Tensor,
    Z_to_index: torch.Tensor,
) -> None:
    if isinstance(epoch, bool) or not isinstance(
        epoch,
        int,
    ):
        raise CheckpointError(
            "epoch must be an integer"
        )

    if epoch < 0:
        raise CheckpointError(
            "epoch must be non-negative"
        )

    best_val_loss = float(best_val_loss)

    if not math.isfinite(best_val_loss):
        raise CheckpointError(
            "best_val_loss must be finite"
        )

    if not isinstance(spinful, bool):
        raise CheckpointError(
            "spinful must be bool"
        )

    for name, tensor in (
        ("index_to_Z", index_to_Z),
        ("Z_to_index", Z_to_index),
    ):
        if not torch.is_tensor(tensor):
            raise CheckpointError(
                f"{name} must be a tensor"
            )

        if tensor.ndim != 1:
            raise CheckpointError(
                f"{name} must be 1-D"
            )

        if tensor.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise CheckpointError(
                f"{name} must have integer dtype"
            )

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    temporary_path = path + ".tmp"

    try:
        with h5py.File(
            temporary_path,
            "w",
        ) as f:
            f.attrs["format"] = FORMAT_NAME
            f.attrs["format_version"] = (
                FORMAT_VERSION
            )

            training_group = f.create_group(
                "training"
            )

            training_group.attrs["epoch"] = (
                epoch
            )

            training_group.attrs[
                "best_val_loss"
            ] = best_val_loss

            dataset_group = f.create_group(
                "dataset"
            )

            dataset_group.attrs["spinful"] = (
                spinful
            )

            _write_tensor(
                dataset_group,
                "index_to_Z",
                index_to_Z,
            )

            _write_tensor(
                dataset_group,
                "Z_to_index",
                Z_to_index,
            )

            model_group = f.create_group(
                "model"
            )

            model_state_group = (
                model_group.create_group(
                    "state_dict"
                )
            )

            _write_model_state(
                model_state_group,
                model,
            )

            if optimizer is not None:
                optimizer_group = (
                    f.create_group(
                        "optimizer"
                    )
                )

                _write_optimizer(
                    optimizer_group,
                    model,
                    optimizer,
                )

            f.flush()

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:
        try:
            if os.path.exists(
                temporary_path
            ):
                os.remove(
                    temporary_path
                )
        except OSError:
            pass

        raise


def _validate_root(
    f: h5py.File,
    *,
    require_optimizer: bool,
) -> None:
    _require(
        "format" in f.attrs,
        "Missing checkpoint format",
    )

    _require(
        _decode_string(
            f.attrs["format"]
        ) == FORMAT_NAME,
        "Not a DeepH HDF5 checkpoint",
    )

    _require(
        "format_version" in f.attrs,
        "Missing checkpoint format version",
    )

    _require(
        int(f.attrs["format_version"])
        == FORMAT_VERSION,
        f"Unsupported checkpoint format version: "
        f"{f.attrs['format_version']}",
    )

    expected_groups = {
        "training",
        "dataset",
        "model",
    }

    if require_optimizer:
        expected_groups.add(
            "optimizer"
        )

    actual_groups = set(
        f.keys()
    )

    if require_optimizer:
        _require(
            actual_groups == expected_groups,
            f"Unexpected checkpoint root objects: "
            f"{sorted(actual_groups)}",
        )
    else:
        _require(
            actual_groups in (
                expected_groups,
                expected_groups
                | {"optimizer"},
            ),
            f"Unexpected checkpoint root objects: "
            f"{sorted(actual_groups)}",
        )


def _read_metadata(
    f: h5py.File,
):
    training_group = f["training"]

    _require(
        isinstance(
            training_group,
            h5py.Group,
        ),
        "training is not a group",
    )

    _require(
        len(training_group) == 0,
        "Unexpected objects under training",
    )

    _require(
        set(training_group.attrs.keys())
        == {
            "epoch",
            "best_val_loss",
        },
        "Unexpected training metadata",
    )

    epoch = int(
        training_group.attrs["epoch"]
    )

    best_val_loss = float(
        training_group.attrs[
            "best_val_loss"
        ]
    )

    _require(
        epoch >= 0,
        "Invalid checkpoint epoch",
    )

    _require(
        math.isfinite(best_val_loss),
        "Invalid best_val_loss",
    )

    dataset_group = f["dataset"]

    _require(
        isinstance(
            dataset_group,
            h5py.Group,
        ),
        "dataset is not a group",
    )

    _require(
        set(dataset_group.keys())
        == {
            "index_to_Z",
            "Z_to_index",
        },
        "Unexpected dataset metadata",
    )

    _require(
        set(dataset_group.attrs.keys())
        == {"spinful"},
        "Unexpected dataset attributes",
    )

    spinful_raw = dataset_group.attrs[
        "spinful"
    ]

    if isinstance(
        spinful_raw,
        (bool, np.bool_),
    ):
        spinful = bool(
            spinful_raw
        )
    else:
        raise CheckpointError(
            "Invalid spinful metadata type"
        )

    index_to_Z = _read_tensor(
        dataset_group[
            "index_to_Z"
        ]
    )

    Z_to_index = _read_tensor(
        dataset_group[
            "Z_to_index"
        ]
    )

    for name, tensor in (
        ("index_to_Z", index_to_Z),
        ("Z_to_index", Z_to_index),
    ):
        _require(
            tensor.ndim == 1,
            f"{name} must be 1-D",
        )

        _require(
            tensor.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ),
            f"{name} must have integer dtype",
        )

    return {
        "epoch": epoch,
        "best_val_loss": (
            best_val_loss
        ),
        "spinful": spinful,
        "index_to_Z": index_to_Z,
        "Z_to_index": Z_to_index,
    }



def read_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Read validated checkpoint metadata without deserializing Python objects."""
    with h5py.File(path, "r") as f:
        _validate_root(f, require_optimizer=False)
        return _read_metadata(f)



def load_pretrained_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Load only shape-compatible tensors for transfer learning."""
    with h5py.File(path, "r") as f:
        _validate_root(f, require_optimizer=False)
        metadata = _read_metadata(f)
        _require(
            set(f["model"].keys()) == {"state_dict"},
            "Unexpected model checkpoint structure",
        )
        transfer_dict = _read_pretrained_model_state(
            f["model"]["state_dict"],
            model,
        )

    transfer_dict = {
        name: tensor.to(device)
        for name, tensor in transfer_dict.items()
    }
    model_dict = model.state_dict()
    for name, tensor in transfer_dict.items():
        model_dict[name] = tensor
        print("Use pretrained parameters:", name)

    model.load_state_dict(model_dict, strict=True)
    return metadata


def load_model_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    with h5py.File(
        path,
        "r",
    ) as f:
        _validate_root(
            f,
            require_optimizer=False,
        )

        metadata = _read_metadata(
            f
        )

        _require(
            set(f["model"].keys())
            == {"state_dict"},
            "Unexpected model checkpoint structure",
        )

        state_dict = _read_model_state(
            f["model"]["state_dict"],
            model,
        )

    state_dict = {
        name: tensor.to(device)
        for name, tensor
        in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return metadata


def load_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    with h5py.File(
        path,
        "r",
    ) as f:
        _validate_root(
            f,
            require_optimizer=True,
        )

        metadata = _read_metadata(
            f
        )

        _require(
            set(f["model"].keys())
            == {"state_dict"},
            "Unexpected model checkpoint structure",
        )

        state_dict = _read_model_state(
            f["model"]["state_dict"],
            model,
        )

        state_dict = {
            name: tensor.to(device)
            for name, tensor
            in state_dict.items()
        }

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        _restore_optimizer(
            f["optimizer"],
            model,
            optimizer,
        )

    return metadata
