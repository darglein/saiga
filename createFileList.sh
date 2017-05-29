#!/bin/bash

FILE="glframework.files"
rm $FILE

echo "glframework.files" >> $FILE
echo "glframework.includes" >> $FILE


for f in $(find src/ -name '*.cpp'); 
do 
echo $f >> $FILE; 
done

for f in $(find src/ -name '*.cu'); 
do 
echo $f >> $FILE; 
done


for f in $(find saiga/ -name '*.h*'); 
do 
echo $f >> $FILE; 
done

for f in $(find shader/ -name '*.glsl'); 
do 
echo $f >> $FILE; 
done
