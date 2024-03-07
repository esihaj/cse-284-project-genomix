import os
import sys
import numpy as np
import pandas as pd
import typer
from typing import Annotated

sys.path.append('./external/gnomix/')
from gnomix import *

app = typer.Typer()

@app.command()
def greet(name: str = typer.Argument(None, help="Your name")):
    """Greets the user with a name, if provided."""
    greeting = f"Hello, {name}" if name else "Hello, World!"
    typer.echo(greeting)

def print_green(text):
    print("\033[92m{}\033[00m" .format(text))

def print_red(text):
    print("\033[91m{}\033[00m" .format(text))
    
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
    print(f"Current working directory: {os.getcwd()}")
    input("Press Enter to continue...")
    
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
        print(f"Running in command line: \n\t{subset_cmd}")
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
    print(f"Running in command line: \n\t{train_cmd}")
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("Error running train command. Exiting...")
        sys.exit(1)
    
    
    

if __name__ == "__main__":
    app()
