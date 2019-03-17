mkdir -p data/datasets
cd data/datasets

# Download Synthetic dataset
for dataset in "PS_Sculpture_Dataset.tgz" "PS_Blobby_Dataset.tgz"; do
    echo "Downloading $dataset"
    wget http://www.visionlab.cs.hku.hk/data/PS-FCN/datasets/$dataset
    tar -xvf $dataset
    rm $dataset
done

# Back to root directory
cd ../../

