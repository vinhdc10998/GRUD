dirname=$(pwd)
result="${dirname%"${dirname##*[!/]}"}" # extglob-free multi-trailing-/ trim
result="${result##*/}"                  # remove everything before the last /
environment_path="$dirname""/environment.yml"
if [ ! -f $environment_path ]
then
echo "--------We will create basic environment!--------"
echo "name: ${result}
dependencies:
 - python=3.8
 - numpy
 - scipy
 - scikit-learn
 - jupyter 
 - notebook 
 - pandas 
 - seaborn
 - pip" > environment.yml
fi
conda env create -f environment.yml