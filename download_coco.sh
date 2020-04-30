if [ ! -d "data/" ];then
  mkdir data
  else
  echo "data/"
fi 
mkdir data/COCO2017
cd data/COCO2017

echo "Prepare to download train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

echo "Prepare to download train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm -f train2017.zip

echo "Prepare to download test2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -f val2017.zip

