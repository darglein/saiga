#!/bin/sh


# vision scenes
wget http://cloud9.cs.fau.de/index.php/s/0fQkKJzGHyVxSGf/download -O vision/data.zip
cd vision
unzip data.zip
rm data.zip
cd ..

# vision bal datasets
wget http://cloud9.cs.fau.de/index.php/s/CIwSqA3RuaDvz38/download -O vision/bal/data.zip
cd vision/bal
unzip data.zip
rm data.zip
cd ../..


