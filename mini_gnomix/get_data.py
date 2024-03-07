import os
import yaml

sys.path.append('./external/gnomix/')
from gnomix import get_data as gnomix_get_data

def _load_and_prepare_config(base_args):
    """
    Load the configuration from a file specified in base_args and prepare it for data retrieval.
    """
    with open(base_args["config_file"], "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)  # Consider using FullLoader for safety

    # Remove 'val' split if its ratio is 0
    val_ratio = config["simulation"]["splits"]["ratios"].get("val")
    if val_ratio is not None and val_ratio == 0:
        del config["simulation"]["splits"]["ratios"]["val"]

    # Add generations configuration if not present
    if not config["simulation"]["splits"].get("gens"):
        generations = config["simulation"]["gens"]
        gens_with_zero = list(set(generations + [0]))
        gens_without_zero = [g for g in generations if g != 0]
        config["simulation"]["splits"]["gens"] = {
            "train1": gens_with_zero,
            "train2": generations,
            "val": gens_without_zero
        }
    
    return config

def get_training_data(base_args):
    """
    Load data using the configuration and base arguments.
    """
    config = _load_and_prepare_config(base_args)
    verbose = config.get("verbose", False)
    data_path = os.path.join(base_args["output_basename"], "generated_data")
    
    return _get_data_using_gnomix(config, data_path, verbose)

def _get_data_using_gnomix(config, data_path, verbose):
    """
    A simplified function that directly calls 'get_data' with the necessary parameters.
    """

    # Assuming 'get_data' is defined elsewhere and expects certain parameters
    # For demonstration, let's consider it needs `data_path` and some config values

    generations = config["simulation"]["splits"]["gens"]
    window_size_cM = config["model"].get("window_size_cM", 1)  # Default value as an example

    # The actual call to `get_data`, adjust according to its signature
    return gnomix_get_data(data_path, generations, window_size_cM, verbose=verbose)
