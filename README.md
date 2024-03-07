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
3. Activate the python virtual env
   ```bash
    python3 -m venv venv
    source venv/bin/activate  # Unix/MacOS
    # or
    venv\Scripts\activate  # Windows
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
mini-gnomix analyze
```

## TODO

- [x] generate simulation data
- [x] read and parse data
- [ ] train model
- [ ] analyze samples 

## License
Inherited From [gnomix](https://github.com/AI-sandbox/gnomix/blob/main/LICENSE.md)

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" must contact the [Stanford Office of Technology Licensing](https://otl.stanford.edu/) for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to this effect.
