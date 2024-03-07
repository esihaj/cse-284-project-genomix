set -eu


function echo_red() {
    echo -e "\033[0;31m$1\033[0m"
}

function echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}

#check if demo/data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz already exists
if [ -f "demo/data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" ]; then
    echo_green "VCF file already exists"
else
    echo_red "Downloading VCF file"
    wget https://hgdownload.cse.ucsc.edu/gbdb/hg19/1000Genomes/phase3/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz -O demo/data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
    echo_green "Downloaded VCF file"
fi

#check if bcftools is already installed
if [ -x "$(command -v bcftools)" ]; then
    echo_green "BCFTools is already installed"
else
    echo_red "Installing BCFTools"
    # https://www.htslib.org/download/
    # https://gist.github.com/adefelicibus/f6fd06df1b4bb104ceeaccdd7325b856
    wget https://github.com/samtools/bcftools/releases/download/1.19/bcftools-1.19.tar.bz2 -O bcftools.tar.bz2
    tar -xjvf bcftools.tar.bz2
    cd bcftools-1.19
    make -j`nproc`
    sudo make prefix=/usr/local/bin install
    sudo ln -s /usr/local/bin/bin/bcftools /usr/bin/bcftools
    echo_green "BCFTools is ready"
fi

# check if the venv exists at dir venv
if [ -d "venv" ]; then
    echo_green "Virtual environment already exists"
else
    echo_red "Creating virtual environment"
    python3 -m venv venv
    echo_green "Virtual environment is ready"
fi

source venv/bin/activate
echo_red "Installing pip requirements.txt"
pip install -r requirements.txt

echo_green "Dependencies installed"