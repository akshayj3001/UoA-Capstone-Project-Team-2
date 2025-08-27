#Creating python environment
#python -m venv vace-env
#source vace-env/bin/activate
python --version

pyenv install 3.10.13
pyenv virtualenv 3.10.13 vace-env
pyenv activate vace-env

#Initiating the python environment via bash script
curl https://pyenv.run | bash
Add: echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

#Installing VACE
sudo apt install -y libffi-dev
sudo apt-get install libsqlite3-dev

pip install packaging (or in requirements.txt)
pip install wheel
sudo apt-get install libffi-dev
sudo apt install nvidia-cuda-toolkit
pip install matplotlib


echo "Installing VACE dependencies"
cd ./VACE
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124  # If PyTorch is not installed.
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1

echo "Moving to models directory"
cd ./models

echo "Downloading VACE model: VACE-Wan2.1-1.3B-Preview"
git clone https://huggingface.co/ali-vilab/VACE-Wan2.1-1.3B-Preview

echo "Moving to VACE directory"
cd ../..
python vace/vace_wan_inference.py --prompt "Hay look at the beautiful mountain and the valley. It is a long river flowing all the way. I can see some cattle grazing."
