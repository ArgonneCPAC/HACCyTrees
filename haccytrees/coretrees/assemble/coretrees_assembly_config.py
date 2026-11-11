import tomli as toml
from dataclasses import dataclass, field
from haccytrees.simulations import Simulation
from typing import List, Tuple


@dataclass
class CoretreesAssemblyConfig:
    simulation: Simulation
    input_base: str
    output_base: str
    split_output: bool = True
    # base path of the per-rank temporary files, or "memory" to keep them in memory
    temporary_path: str = "./tmp"

    node_position_x: str = "fof_halo_center_x"
    node_position_y: str = "fof_halo_center_y"
    node_position_z: str = "fof_halo_center_z"

    copy_fields: List[Tuple[str]] = field(default_factory=list)
    verbose: int = 0

    # compression of the forest datasets (haccytrees.utils.hdf5_utils): "none" (default,
    # as before), "gzip", "blosc2-zstd" or "blosc2-lz4"
    compression: str = "none"
    compression_level: int = 1

    @classmethod
    def parse_config(cls, config_path: str) -> "CoretreesAssemblyConfig":
        with open(config_path, "rb") as fp:
            config = toml.load(fp)
        if config["simulation"]["simulation"][-4:] == ".cfg":
            simulation = Simulation.parse_config(config["simulation"]["simulation"])
        else:
            simulation = Simulation.simulations[config["simulation"]["simulation"]]

        return cls(
            simulation=simulation,
            input_base=config["simulation"]["coreproperties_base"],
            output_base=config["output"]["output_base"],
            split_output=config["output"]["split_output"],
            temporary_path=config["output"]["temporary_path"],
            node_position_x=config["columns"]["node_position_x"],
            node_position_y=config["columns"]["node_position_y"],
            node_position_z=config["columns"]["node_position_z"],
            copy_fields=config["output"]["copy_fields"],
            verbose=config["algorithm"]["verbose"],
            compression=config["output"].get("compression", "none"),
            compression_level=int(config["output"].get("compression_level", 1)),
        )
