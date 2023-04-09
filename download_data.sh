# Taken from https://github.com/Kai-46/ARF-svox2/blob/master/download_data.sh

pip install gdown

mkdir -p data
cd data

gdown "16VnMcF1KJYxN9QId6TClMsZRahHNMW5g&confirm=t"
unzip nerf_llff_data.zip && mv nerf_llff_data llff

gdown "1PG-KllCv4vSRPO7n5lpBjyTjlUyT8Nag&confirm=t"
tar -xvf lego_real_night_radial.tar.gz && mkdir -p custom  && mv lego_real_night_radial custom/lego

gdown "10Tj-0uh_zIIXf0FZ6vT7_te90VsDnfCU&confirm=t"
unzip TanksAndTempleBG.zip && mv TanksAndTempleBG tnt

gdown "19HLCSEwnfN_Bim3A-OfzygF6qCIeIIdF&confirm=t"
unzip styles.zip

cd ..