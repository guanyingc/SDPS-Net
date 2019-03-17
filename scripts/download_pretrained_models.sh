path="data/models/"
mkdir -p $path
cd $path

# Download pre-trained model
for model in "LCNet_CVPR2019.pth.tar" "NENet_CVPR2019.pth.tar"; do
    wget http://www.visionlab.cs.hku.hk/data/SDPS-Net/models/${model}
done

# Back to root directory
cd ../
