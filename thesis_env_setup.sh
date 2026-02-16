
printf "\n\n[INFO] : SOURCING CONDA ENV\n" 
source /opt/miniconda3/etc/profile.d/conda.sh

printf "\n\n[INFO] : ACTIVATING ENV 'thesis' and installing torch\n"
conda activate thesis && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


printf "\n\n[INFO] : SETTING UP SAM3 DEPENDENCIES\n"
cd sam3 && pip install -e . && cd ..


printf "\n\n[INFO] : INSTALLING 'ipykernel' AND 'ipywidgets'\n"
pip install ipykernel ipywidgets

printf "\n\n[INFO] : CREATING KERNEL 'thesis' AND LINKING TO CONDA ENV 'thesis'\n"
python -m ipykernel install --user --name thesis --display-name "thesis"

printf "\n\n[INFO] : INSTALLING 'opencv', 'matplotlib', 'tqdm' and 'scikit-learn'\n"
pip install opencv-python --no-deps && pip install matplotlib && pip install tqdm && pip install scikit-learn


printf "\n\n[INFO] : INSTALLING 'transformers' from git\n"
pip install git+https://github.com/huggingface/transformers.git


printf "\n\n[INFO] : KERNEL 'thesis' READY FOR USE\n"
