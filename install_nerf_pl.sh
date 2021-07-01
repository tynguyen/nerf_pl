git clone https://github.com/tynguyen/neural_radiance_fields.git
cd nerf_pl

# Install pre-requisites
chmod a+x install_prerequisites.sh
sudo ./install_prerequisites.sh

# Create a conda env
conda create -n nerf -y python=3.7
conda activate nerf

# Install python-related
pip install -r requirements.txt

# Install torchsearchsorted
cd torchsearchsorted
pip install  .
