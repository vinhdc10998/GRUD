read -n 1 -p "Will export environment? [Y/n]: " export_env
if [ "$export_env" != "${export_env#[Yy]}" ] ;then
    conda env export > environment.yml
fi
dirname=$(pwd)
result="${dirname%"${dirname##*[!/]}"}" # extglob-free multi-trailing-/ trim
result="${result##*/}"                  # remove everything before the last /
conda env remove -n ${result}