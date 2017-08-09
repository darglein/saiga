#!/bin/bash

FILE="saiga.files"
ROOTDIR=".."
rm $FILE

echo "saiga.files" >> $FILE
echo "saiga.includes" >> $FILE

for f in $(find ${ROOTDIR}/samples/ -name '*.cpp'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/samples/ -name '*.h'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/src/ -name '*.cpp'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/src/ -name '*.cu'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/include/saiga/ -name '*.h*'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/include/saiga/ -name '*.inl*'); 
do 
echo $f >> $FILE; 
done

for f in $(find ${ROOTDIR}/shader/ -name '*.glsl'); 
do 
echo $f >> $FILE; 
done
