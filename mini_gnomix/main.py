import os
import sys
import numpy as np
import pandas as pd
import typer
import shutil
from typing import Annotated

import gzip
import numpy as np
import os
import pandas as pd
import pickle
import sys
import yaml

from .get_data import get_training_data
from mini_gnomix.train import Gnomix

sys.path.append('./external/gnomix/')

from src.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, npy_to_vcf, update_vcf 
from src.utils import read_genetic_map, save_dict, load_dict, read_headers
from src.preprocess import load_np_data, data_process
from src.postprocess import get_meta_data, write_msp, write_fb, msp_to_lai, msp_to_bed
from src.visualization import plot_cm, plot_chm
from src.laidataset import LAIDataset



app = typer.Typer()

def print_green(text):
    print("\033[92m{}\033[00m" .format(text))

def print_red(text):
    print("\033[91m{}\033[00m" .format(text))
    
def print_yellow(text):
    print("\033[93m{}\033[00m" .format(text))
    
@app.command()
def analyze(query_file: Annotated[str, typer.Argument()] = "small_query_chr22.vcf.gz",
            model_path: Annotated[str, typer.Argument()] = "./data/output/model"):
    """Perform local ancestry inference on a query file """
    print_green("Analyzing the query file...")
    print("Not yet implemented!")

@app.command()
def train():

    # data_path contains - train1/, train2/, val/, metadata, sample_maps/

    verbose = True
    
    base_args = {
        'output_basename': './demo/output',
        'chm': '22',
        'config_file': "./config.yaml"
    }

    with open(base_args["config_file"],"r") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    data_path = config["simulation"]["path"]
    print(f" data path is: {data_path}")

    rm_simulated_data=config["simulation"]["rm_data"]
    model_name=config["model"].get("name", "model")
    inference=config["model"].get("inference", "default")
    window_size_cM=config["model"].get("window_size_cM")
    smooth_window_size=config["model"].get("smooth_size")
    n_cores=config["model"].get("n_cores", None)
    retrain_base=config["model"].get("retrain_base")
    calibrate=config["model"].get("calibrate")
    context_ratio=config["model"].get("context_ratio")
    chm = base_args["chm"]

    # option to bypass validation
    ratios = config["simulation"]["splits"]["ratios"]
    validate = True if ratios.get("val") else False

    # generations = config["simulation"]["splits"]["gens"]
    # if validate == False:
    #     del generations["val"]

    output_path = base_args["output_basename"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Either load pre-trained model or simulate data from reference file, init model and train it
    # Processing data
    
    if verbose:
        print_green("Loading data...")
    
    
    data, meta = get_training_data(base_args)
    if verbose:
        print_green ("Loaded data successfully!")

    print("meta: ", meta, "\n")
    for i, group in enumerate(data):
        print(f"Split {i+1}:")
        for j, array in enumerate(group):
            label = "SNPs" if j == 0 else "Ancestry windows" if j == 1 else f"Array {j+1}"
            shape = f"({array.shape[0]:>5}, {array.shape[1]:>7})"  # Right-align numbers within specified width
            dtype = f"{array.dtype}"
            print(f"  {label:17}: {shape} - {dtype}")

    print_green("initializing model")
    

    # init model
    model = Gnomix(C=meta["C"], M=meta["M"], A=meta["A"], S=smooth_window_size,
                    snp_pos=meta["snp_pos"], snp_ref=meta["snp_ref"], snp_alt=meta["snp_alt"],
                    population_order=meta["pop_order"],
                    mode=inference, calibrate=calibrate,
                    n_jobs=n_cores, context_ratio=context_ratio, seed=config["seed"])

    # train it
    if verbose:
        print("Building model...")
    model.train(data=data, retrain_base=retrain_base, evaluate=True, verbose=verbose)
    # write gentic map df
    model.write_gen_map_df(load_dict(os.path.join(data_path,"gen_map_df.pkl")))

    # store it
    model_repo = join_paths(output_path, "models", verb=False)
    model_repo = join_paths(model_repo, model_name + "_chm_" + str(chm), verb=False)
    model_path = model_repo + "/" + model_name + "_chm_" + str(chm) + ".pkl"
    pickle.dump(model, open(model_path,"wb"))

    # brief analysis
    if verbose:
        print("Analyzing model performance...")
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    cm_path = analysis_path+"/confusion_matrix_{}.txt"
    cm_plot_path = analysis_path+"/confusion_matrix_{}_normalized.png"
    analysis_sets = ["train", "val"] if validate else ["train"]
    for d in analysis_sets:
        cm, idx = model.Confusion_Matrices[d]
        n_digits = int(np.ceil(np.log10(np.max(cm))))
        np.savetxt(cm_path.format(d), cm, fmt='%-'+str(n_digits)+'.0f')
        plot_cm(cm, labels=model.population_order[idx], path=cm_plot_path.format(d))
        if verbose:
            print("Estimated "+d+" accuracy: {}%".format(model.accuracies["smooth_"+d+"_acc"]))

    # write the model parameters of type int, float, str into a file config TODO: test
    model_config_path = os.path.join(model_repo, "config.txt")
    model.write_config(model_config_path)

    if verbose:
        print("Model, info and analysis saved at {}".format(model_repo))
        print("-"*80)

    if rm_simulated_data:
        if verbose:
            print("Removing simulated data...")
        splits_to_rem = ["train1","train2","val"] if validate else ["train1","train2"]
        for split in splits_to_rem: # train1, train2, val (if val is there)
            chm_path = join_paths(data_path, split, verb=False)
            remove_data_cmd = "rm -r " + chm_path
            run_shell_cmd(remove_data_cmd, verbose=False)

    return model
 
# @app.command()
# def train(name: str = typer.Argument(None, help="Your name")):
#     """Load the data and train the model."""
#     print_green("Loading data...")
    
#     base_args = {
#         'output_basename': './demo/output',
#         'chm': '22',
#         'config_file': "./config.yaml"
#     }
#     data, meta = get_training_data(base_args)
#     print_green ("Loaded data successfully!")
    
#     print("meta: ", meta, "\n")
#     for i, group in enumerate(data):
#         print(f"Split {i+1}:")
#         for j, array in enumerate(group):
#             label = "SNPs" if j == 0 else "Ancestry windows" if j == 1 else f"Array {j+1}"
#             shape = f"({array.shape[0]:>5}, {array.shape[1]:>7})"  # Right-align numbers within specified width
#             dtype = f"{array.dtype}"
#             print(f"  {label:17}: {shape} - {dtype}")
      
@app.command()
def simulate_data(data_path: Annotated[str, typer.Argument()] = "./demo/data/",
         query_file: Annotated[str, typer.Argument()] = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz",
         genetic_map_file: Annotated[str, typer.Argument()] = "allchrs.b37.gmap",
         reference_file: Annotated[str, typer.Argument()] = "reference_1000g.vcf",
         sample_map_file: Annotated[str, typer.Argument()] = "1000g.smap",
         chm: Annotated[str, typer.Argument()] = "22",
         phase: Annotated[bool, typer.Option("--phased", help="Use phased data")] = False,
         output_basename: Annotated[str, typer.Argument()] = "./demo/output"):
    """Simulated Admixed Training Data Using Gnomix"""
    # This is based on external/gnomix/demo.ipynb
    print(f"Current working directory: {os.getcwd()}")
    
    query_file = os.path.join(data_path, query_file)
    genetic_map_file = os.path.join(data_path, genetic_map_file)
    reference_file = os.path.join(data_path, reference_file)
    sample_map_file = os.path.join(data_path, sample_map_file)

    # Generate reference file
    #check if reference file exists
    if not os.path.exists(reference_file):
        print_green("Generating reference file...")
        print_red("This will take some time!")
        sample_map = pd.read_csv(sample_map_file, sep="\t")
        samples = list(sample_map["#Sample"])
        sample_file = os.path.join(data_path, "samples_1000g.tsv")
        np.savetxt(sample_file, samples, delimiter="\t", fmt="%s")
        subset_cmd = f"bcftools view -S {sample_file} -o {reference_file} {query_file}"
        print_yellow(f"Running in command line: \n\t{subset_cmd}")
        exit_code = os.system(subset_cmd)
        if exit_code != 0:
            print_red("Error running subset command. Exiting...")
            sys.exit(1)
        print_green("Reference file generated successfully!")
    else:
        print_green("Reference file already exists.")
    

    # Simulate the data and train the model
    print_green("Simulating data and training the model...")
    train_cmd = f"python3 external/gnomix/gnomix.py {query_file} {output_basename} {chm} {'True' if phase else 'False'} {genetic_map_file} {reference_file} {sample_map_file}"
    # > ./demo/training_log.txt
    print_red("Remember that you just need to wait until simulation data is done! You don't need to wait for the training process")
    print_red("This will take some time!")
    # copy the config.yaml file from "external/gnomix" to here 
    shutil.copy('./external/gnomix/config.yaml', './')
    print_yellow(f"copied config.yaml file from external/gnomix to here. You may delete it later.")
    print_yellow(f"Running in command line: \n\t{train_cmd}")
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("Error running train command. Exiting...")
        sys.exit(1)
    
    
    

if __name__ == "__main__":
    app()
