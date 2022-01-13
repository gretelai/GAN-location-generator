URL=https://gretel-public-website.s3.amazonaws.com/datasets/ebike_locations.tar.gz
TAR_FILE=./datasets/ebike_locations.tar.gz
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
