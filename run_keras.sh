#!/bin/bash

echo "Loading modules."

module load anaconda3 CUDA cuDNN

echo "Done."

ENV="capsenv"

echo "Starting conda enviroment '$ENV'"

echo $PWD
source activate $ENV
if [ $? -eq 0 ]; then
	:
else
	echo "Conda env '$ENV' doesn't exist."
	echo "Creating enviroment '$ENV':"
	conda env create --name $ENV --file keras.yml 
	source activate $ENV
fi

#echo "Done."
#echo "Starting jupyter notebook."

#jupyter notebook --port=3333
