#!/bin/sh

# extra models
cd models
wget http://cloud9.cs.fau.de/index.php/s/ErxcdZJu4aXuLtO/download -O data.zip
unzip data.zip
rm data.zip
cd ..

exit

# vision scenes
cd vision
wget http://cloud9.cs.fau.de/index.php/s/0fQkKJzGHyVxSGf/download -O data.zip
unzip data.zip
rm data.zip
cd ..

# vision bal datasets
cd vision/bal
wget http://cloud9.cs.fau.de/index.php/s/CIwSqA3RuaDvz38/download -O data.zip
unzip data.zip
rm data.zip
cd ../..



