python --version
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