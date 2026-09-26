import os
import time
import tqdm

import h5py
import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool as Pool
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, InMemoryDataset

from .graph import get_graph


HGRAPH_FORMAT = "deeph-hgraph"
HGRAPH_FORMAT_VERSION = 1

_REQUIRED_GRAPH_TENSORS = {
    "atom_num_orbital",
    "x",
    "edge_index",
    "edge_attr",
    "term_mask",
}

_OPTIONAL_GRAPH_TENSORS = {
    "term_real",
    "rh",
    "rdm",
    "rvdee",
    "rvxc",
    "rvna",
    "label",
    "mask",
}

_REQUIRED_SUBGRAPH_TENSORS = {
    "subgraph_atom_idx",
    "subgraph_edge_idx",
    "subgraph_edge_ang",
    "subgraph_index",
}


def _tensor_to_numpy(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype == torch.bfloat16:
        raise TypeError("bfloat16 graph serialization is not supported")

    return tensor.numpy()


def _write_tensor(group, name, tensor):
    array = _tensor_to_numpy(tensor)
    group.create_dataset(name, data=array)


def _read_tensor(group, name):
    if name not in group:
        raise RuntimeError(f"Missing graph tensor: {group.name}/{name}")

    obj = group[name]

    if not isinstance(obj, h5py.Dataset):
        raise RuntimeError(
            f"Expected HDF5 dataset at {obj.name}, got {type(obj)!r}"
        )

    array = np.asarray(obj)

    if array.dtype.kind not in ("b", "i", "u", "f", "c"):
        raise RuntimeError(
            f"Unsupported dtype {array.dtype} at {obj.name}"
        )

    return torch.from_numpy(array.copy())


def _decode_hdf5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, str):
        return value

    raise RuntimeError(
        f"Expected HDF5 string attribute, got {type(value)!r}"
    )


class HData(InMemoryDataset):
    def __init__(
        self,
        raw_data_dir: str,
        graph_dir: str,
        interface: str,
        target: str,
        dataset_name: str,
        multiprocessing: int,
        radius,
        max_num_nbr,
        num_l,
        max_element,
        create_from_DFT,
        if_lcmp_graph,
        separate_onsite,
        new_sp,
        default_dtype_torch,
        nums: int = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        When interface == 'h5',

        raw_data_dir
        ├── 00
        │   ├── rh.h5 / rdm.h5
        │   ├── rc.h5
        │   ├── element.dat
        │   ├── orbital_types.dat
        │   ├── site_positions.dat
        │   ├── lat.dat
        │   └── info.json
        ├── 01
        │   └── ...
        └── ...

        Processed graph caches are stored as structured HDF5.

        No pickle deserialization is used for graph caches.
        """

        self.raw_data_dir = raw_data_dir

        assert "-" not in dataset_name, (
            '"-" can not be included in the dataset name'
        )

        if create_from_DFT:
            way_create_graph = "FromDFT"
        else:
            way_create_graph = f"{radius}r{max_num_nbr}mn"

        if if_lcmp_graph:
            lcmp_str = f"{num_l}l"
        else:
            lcmp_str = "WithoutLCMP"

        if separate_onsite is True:
            onsite_str = "-SeparateOnsite"
        else:
            onsite_str = ""

        if new_sp:
            new_sp_str = "-NewSP"
        else:
            new_sp_str = ""

        if target == "hamiltonian":
            title = "HGraph"
        else:
            raise ValueError(
                "Unknown prediction target: {}".format(target)
            )

        graph_file_name = (
            f"{title}-{interface}-{dataset_name}-{lcmp_str}-"
            f"{way_create_graph}{onsite_str}{new_sp_str}.h5"
        )

        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)

        self.data = None
        self.slices = None

        self.interface = interface
        self.target = target
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.num_l = num_l
        self.create_from_DFT = create_from_DFT
        self.if_lcmp_graph = if_lcmp_graph
        self.separate_onsite = separate_onsite
        self.new_sp = new_sp
        self.default_dtype_torch = default_dtype_torch

        self.nums = nums
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        print(f"Graph data file: {graph_file_name}")

        if os.path.exists(self.data_file):
            print("Use existing graph data file")
        else:
            print("Process new data file......")
            self.process()

        begin = time.time()

        data_list, self.info = self._load_hdf5()

        if len(data_list) == 0:
            raise RuntimeError("Graph cache contains no structures")

        self.data, self.slices = self.collate(data_list)

        print(
            f"Atomic types: {self.info['index_to_Z'].tolist()}"
        )

        print(
            f"Finish loading the processed {len(self)} structures "
            f"(spinful: {self.info['spinful']}, "
            f"the number of atomic types: "
            f"{len(self.info['index_to_Z'])}), "
            f"cost {time.time() - begin:.0f} seconds"
        )

    def _validate_graph(self, graph):
        if not isinstance(graph, Data):
            raise TypeError(
                f"Expected torch_geometric.data.Data, "
                f"got {type(graph)!r}"
            )

        keys = set(graph.keys())

        missing = _REQUIRED_GRAPH_TENSORS - keys
        if missing:
            raise RuntimeError(
                f"Graph is missing required fields: {sorted(missing)}"
            )

        for key in _REQUIRED_GRAPH_TENSORS:
            if not torch.is_tensor(graph[key]):
                raise TypeError(
                    f"Graph field {key!r} must be a tensor"
                )

        if graph.x.dtype != torch.int64:
            raise RuntimeError(
                f"graph.x must be torch.int64, got {graph.x.dtype}"
            )

        if graph.edge_index.dtype != torch.int64:
            raise RuntimeError(
                "graph.edge_index must be torch.int64, "
                f"got {graph.edge_index.dtype}"
            )

        if graph.edge_index.ndim != 2:
            raise RuntimeError(
                "graph.edge_index must be rank 2"
            )

        if graph.edge_index.shape[0] != 2:
            raise RuntimeError(
                "graph.edge_index must have shape (2, num_edges)"
            )

        if graph.edge_attr.ndim < 1:
            raise RuntimeError(
                "graph.edge_attr must have at least one dimension"
            )

        num_edges = graph.edge_index.shape[1]

        if graph.edge_attr.shape[0] != num_edges:
            raise RuntimeError(
                "edge_attr and edge_index contain different "
                "numbers of edges"
            )

        if graph.term_mask.ndim != 1:
            raise RuntimeError(
                "term_mask must be rank 1"
            )

        if graph.term_mask.shape[0] != num_edges:
            raise RuntimeError(
                "term_mask and edge_index contain different "
                "numbers of edges"
            )

        if graph.term_mask.dtype != torch.bool:
            raise RuntimeError(
                "term_mask must have dtype torch.bool"
            )

        if not hasattr(graph, "stru_id"):
            raise RuntimeError("Graph is missing stru_id")

        if not isinstance(graph.stru_id, str):
            raise TypeError("Graph stru_id must be str")

        if not hasattr(graph, "spinful"):
            raise RuntimeError("Graph is missing spinful")

        if not isinstance(graph.spinful, (bool, np.bool_)):
            raise TypeError("Graph spinful must be bool")

        if self.if_lcmp_graph:
            if not hasattr(graph, "subgraph_dict"):
                raise RuntimeError(
                    "LCMP graph is missing subgraph_dict"
                )

            if not isinstance(graph.subgraph_dict, dict):
                raise TypeError(
                    "subgraph_dict must be dict"
                )

            subgraph_keys = set(graph.subgraph_dict.keys())

            if subgraph_keys != _REQUIRED_SUBGRAPH_TENSORS:
                raise RuntimeError(
                    "Unexpected subgraph_dict keys: "
                    f"{sorted(subgraph_keys)}"
                )

            for key in _REQUIRED_SUBGRAPH_TENSORS:
                value = graph.subgraph_dict[key]

                if not torch.is_tensor(value):
                    raise TypeError(
                        f"subgraph_dict[{key!r}] must be tensor"
                    )

    def _save_hdf5(self, data_list, info):
        temporary_file = self.data_file + ".tmp"

        if os.path.exists(temporary_file):
            os.remove(temporary_file)

        try:
            with h5py.File(temporary_file, "w") as f:
                f.attrs["format"] = HGRAPH_FORMAT
                f.attrs["format_version"] = HGRAPH_FORMAT_VERSION
                f.attrs["num_graphs"] = len(data_list)
                f.attrs["spinful"] = bool(info["spinful"])

                info_group = f.create_group("info")

                _write_tensor(
                    info_group,
                    "index_to_Z",
                    info["index_to_Z"],
                )

                _write_tensor(
                    info_group,
                    "Z_to_index",
                    info["Z_to_index"],
                )

                graphs_group = f.create_group("graphs")

                for index, graph in enumerate(data_list):
                    self._validate_graph(graph)

                    group = graphs_group.create_group(
                        f"{index:08d}"
                    )

                    group.attrs["stru_id"] = graph.stru_id
                    group.attrs["spinful"] = bool(graph.spinful)

                    for key in graph.keys():
                        value = graph[key]

                        if torch.is_tensor(value):
                            _write_tensor(group, key, value)

                    if hasattr(graph, "subgraph_dict"):
                        subgraph_group = group.create_group(
                            "subgraph_dict"
                        )

                        for key in (
                            "subgraph_atom_idx",
                            "subgraph_edge_idx",
                            "subgraph_edge_ang",
                            "subgraph_index",
                        ):
                            _write_tensor(
                                subgraph_group,
                                key,
                                graph.subgraph_dict[key],
                            )

                f.flush()

            os.replace(temporary_file, self.data_file)

        except Exception:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
            raise

    def _load_hdf5(self):
        data_list = []

        with h5py.File(self.data_file, "r") as f:
            file_format = _decode_hdf5_string(
                f.attrs.get("format", "")
            )

            if file_format != HGRAPH_FORMAT:
                raise RuntimeError(
                    f"Invalid graph cache format: {file_format!r}"
                )

            version = int(
                f.attrs.get("format_version", -1)
            )

            if version != HGRAPH_FORMAT_VERSION:
                raise RuntimeError(
                    "Unsupported graph cache format version: "
                    f"{version}"
                )

            if "info" not in f:
                raise RuntimeError(
                    "Graph cache is missing /info"
                )

            if "graphs" not in f:
                raise RuntimeError(
                    "Graph cache is missing /graphs"
                )

            info_group = f["info"]

            index_to_Z = _read_tensor(
                info_group,
                "index_to_Z",
            ).to(torch.int64)

            Z_to_index = _read_tensor(
                info_group,
                "Z_to_index",
            ).to(torch.int64)

            spinful = bool(
                f.attrs.get("spinful", False)
            )

            graphs_group = f["graphs"]

            graph_names = sorted(graphs_group.keys())

            expected_num_graphs = int(
                f.attrs.get("num_graphs", -1)
            )

            if expected_num_graphs != len(graph_names):
                raise RuntimeError(
                    "Graph cache num_graphs does not match "
                    "the number of graph groups"
                )

            for graph_name in graph_names:
                group = graphs_group[graph_name]

                if not isinstance(group, h5py.Group):
                    raise RuntimeError(
                        f"{group.name} is not an HDF5 group"
                    )

                stru_id = _decode_hdf5_string(
                    group.attrs.get("stru_id", "")
                )

                graph_spinful = bool(
                    group.attrs.get("spinful", False)
                )

                kwargs = {}

                for key in group.keys():
                    if key == "subgraph_dict":
                        continue

                    obj = group[key]

                    if not isinstance(obj, h5py.Dataset):
                        raise RuntimeError(
                            f"Unexpected HDF5 object at {obj.name}"
                        )

                    kwargs[key] = _read_tensor(
                        group,
                        key,
                    )

                kwargs["stru_id"] = stru_id
                kwargs["spinful"] = graph_spinful

                if "subgraph_dict" in group:
                    subgraph_group = group["subgraph_dict"]

                    if not isinstance(
                        subgraph_group,
                        h5py.Group,
                    ):
                        raise RuntimeError(
                            "subgraph_dict is not an HDF5 group"
                        )

                    actual_keys = set(
                        subgraph_group.keys()
                    )

                    if (
                        actual_keys
                        != _REQUIRED_SUBGRAPH_TENSORS
                    ):
                        raise RuntimeError(
                            "Invalid subgraph_dict keys: "
                            f"{sorted(actual_keys)}"
                        )

                    kwargs["subgraph_dict"] = {
                        key: _read_tensor(
                            subgraph_group,
                            key,
                        )
                        for key in (
                            "subgraph_atom_idx",
                            "subgraph_edge_idx",
                            "subgraph_edge_ang",
                            "subgraph_index",
                        )
                    }

                graph = Data(**kwargs)

                self._validate_graph(graph)

                if bool(graph.spinful) != spinful:
                    raise RuntimeError(
                        "Inconsistent spinful flag in graph cache"
                    )

                data_list.append(graph)

        info = {
            "spinful": spinful,
            "index_to_Z": index_to_Z,
            "Z_to_index": Z_to_index,
        }

        return data_list, info

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]

        structure = Structure(
            np.loadtxt(
                os.path.join(folder, "lat.dat")
            ).T,
            np.loadtxt(
                os.path.join(folder, "element.dat")
            ),
            np.loadtxt(
                os.path.join(
                    folder,
                    "site_positions.dat",
                )
            ).T,
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

        cart_coords = torch.tensor(
            structure.cart_coords,
            dtype=self.default_dtype_torch,
        )

        frac_coords = torch.tensor(
            structure.frac_coords,
            dtype=self.default_dtype_torch,
        )

        numbers = torch.tensor(
            structure.atomic_numbers
        )

        structure.lattice.matrix.setflags(
            write=True
        )

        lattice = torch.tensor(
            structure.lattice.matrix,
            dtype=self.default_dtype_torch,
        )

        if self.target == "E_ij":
            huge_structure = True
        else:
            huge_structure = False

        return get_graph(
            cart_coords,
            frac_coords,
            numbers,
            stru_id,
            r=self.radius,
            max_num_nbr=self.max_num_nbr,
            numerical_tol=1e-8,
            lattice=lattice,
            default_dtype_torch=self.default_dtype_torch,
            tb_folder=folder,
            interface=self.interface,
            num_l=self.num_l,
            create_from_DFT=self.create_from_DFT,
            if_lcmp_graph=self.if_lcmp_graph,
            separate_onsite=self.separate_onsite,
            target=self.target,
            huge_structure=huge_structure,
            if_new_sp=self.new_sp,
            **kwargs,
        )

    def process(self):
        begin = time.time()

        folder_list = []

        for root, dirs, files in os.walk(
            self.raw_data_dir
        ):
            if (
                self.interface == "h5"
                and "rc.h5" in files
            ) or (
                self.interface == "npz"
                and "rc.npz" in files
            ):
                folder_list.append(root)

        folder_list = sorted(folder_list)
        folder_list = folder_list[: self.nums]

        if self.dataset_name == "graphene_450":
            folder_list = folder_list[500:5000:10]

        if self.dataset_name == "graphene_1500":
            folder_list = folder_list[500:5000:3]

        if self.dataset_name == "bp_bilayer":
            folder_list = folder_list[:600]

        assert len(folder_list) != 0, (
            "Can not find any structure"
        )

        print(
            "Found %d structures, have cost %d seconds"
            % (
                len(folder_list),
                time.time() - begin,
            )
        )

        if self.multiprocessing == 0:
            print(
                "Use multiprocessing "
                "(nodes = num_processors x num_threads "
                f"= 1 x {torch.get_num_threads()})"
            )

            data_list = [
                self.process_worker(folder)
                for folder in tqdm.tqdm(folder_list)
            ]

        else:
            pool_dict = (
                {}
                if self.multiprocessing < 0
                else {"nodes": self.multiprocessing}
            )

            torch_num_threads = (
                torch.get_num_threads()
            )

            torch.set_num_threads(1)

            with Pool(**pool_dict) as pool:
                nodes = pool.nodes

                print(
                    "Use multiprocessing "
                    "(nodes = num_processors x num_threads "
                    f"= {nodes} x "
                    f"{torch.get_num_threads()})"
                )

                data_list = list(
                    tqdm.tqdm(
                        pool.imap(
                            self.process_worker,
                            folder_list,
                        ),
                        total=len(folder_list),
                    )
                )

            torch.set_num_threads(
                torch_num_threads
            )

        print(
            "Finish processing %d structures, "
            "have cost %d seconds"
            % (
                len(data_list),
                time.time() - begin,
            )
        )

        if self.pre_filter is not None:
            data_list = [
                d
                for d in data_list
                if self.pre_filter(d)
            ]

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(d)
                for d in data_list
            ]

        index_to_Z, Z_to_index = (
            self.element_statistics(data_list)
        )

        spinful = data_list[0].spinful

        for data in data_list:
            if bool(data.spinful) != bool(spinful):
                raise RuntimeError(
                    "Structures have inconsistent spinful flags"
                )

        info = {
            "spinful": bool(spinful),
            "index_to_Z": index_to_Z,
            "Z_to_index": Z_to_index,
        }

        self._save_hdf5(
            data_list,
            info,
        )

        print(
            "Finish saving %d structures to %s, "
            "have cost %d seconds"
            % (
                len(data_list),
                self.data_file,
                time.time() - begin,
            )
        )

    def element_statistics(self, data_list):
        index_to_Z, inverse_indices = torch.unique(
            data_list[0].x,
            sorted=True,
            return_inverse=True,
        )

        Z_to_index = torch.full(
            (100,),
            -1,
            dtype=torch.int64,
        )

        Z_to_index[index_to_Z] = torch.arange(
            len(index_to_Z)
        )

        for data in data_list:
            data.x = Z_to_index[data.x]

        return index_to_Z, Z_to_index
