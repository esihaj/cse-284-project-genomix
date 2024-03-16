# CSE-284 Project: Mini-Gnomix CLI Tool
## Group 
Group Number 2, Members: Ehsan Hajyasini & Andre Wang


## Introduction

Mini-Gnomix is a re-implementation of [AI-sandbox/gnomix](https://github.com/AI-sandbox/gnomix) as the course project of CSE 284. Mini-Genomix is a tool to perform Local Ancestry Inference. In our implementation we first use gnomix to simulate SNP and LAI data based on a set of non-admixed founder individuals (extracted from 1000 Genomes project). Then we train a logistic regression to perform the classification, followed by XGBoost to perform smoothing.

## Installation

To install Mini-Gnomix, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone --recursive https://github.com/esihaj/cse-284-project-genomix.git
    cd cse-284-project-genomix
    ```
2. Install the dependencies
   ```bash
   ./dependencies.sh
   ```
This file downloads the 1000 Genome chromosome 22 data, installs bcftools, sets up a python venv and install pip requirements.txt.

2.1 If you encounter any problem here because of the wget from Github due to the proxy status of your server, you can choose to download the file into your local computer and bypass this command
```
wget https://github.com/samtools/bcftools/releases/download/1.19/bcftools-1.19.tar.bz2 -O bcftools.tar.bz2
```
You then need to manually run the following commands
```tar -xjvf bcftools.tar.bz2
    cd bcftools-1.19
    make -j`nproc`
    sudo make prefix=/usr/local/bin install
    sudo ln -s /usr/local/bin/bin/bcftools /usr/bin/bcftools
```
If you don't have sudo access, try the following lines to create symbolic link and bypass the above last two lines
```
make prefix=~/local install
export PATH="$HOME/local/bin:$PATH"
source ~/.bashrc
```
Then, rerun the entire dependencies.sh
   
3. Activate the python virtual env
   ```bash
    source venv/bin/activate  # Unix/MacOS
    ```
4. Install Mini-Gnomix:

    ```bash
    pip install --editable .
    ```

## Usage

To use Mini-Gnomix, run the following command:

```bash
mini-gnomix --help # to list all the commands
mini-gnomix simulate-data
mini-gnomix train
```

Please be aware that the simulate-data command may display error messages upon completion. These are typically caused by mismatches in dependency versions but occur only during the final stage of saving configuration files with gnomix. These errors do not affect the operation's outcome and can be safely disregarded.

## TODO

- [x] generate simulation data
- [x] read and parse data
- [x] train model
- [x] analyze samples 

## License
Inherited From [gnomix](https://github.com/AI-sandbox/gnomix/blob/main/LICENSE.md)

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" must contact the [Stanford Office of Technology Licensing](https://otl.stanford.edu/) for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to this effect.
