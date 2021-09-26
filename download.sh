mkdir -p minicore_data
curl -s https://zenodo.org/record/4738365/files/1M.tar.xz?download=1 > minicore_data/1M.tar.xz
curl -s https://zenodo.org/record/4738365/files/293T.tar.xz?download=1 > minicore_data/293T.tar.xz
curl -s https://zenodo.org/record/4738365/files/CAO.tar.xz?download=1 > minicore_data/CAO.tar.xz
curl -s https://zenodo.org/record/4738365/files/PBMC.tar.xz?download=1 > minicore_data/PBMC.tar.xz
curl -s https://zenodo.org/record/4738365/files/ZEISEL.tar.xz?download=1 > minicore_data/ZEISEL.tar.xz
cd minicore_data
for f in *.tar.xz
do
    tar -Jxf $f && rm $f
done

# To add to your bashrc:
while true; do
read -p "Do you wish to add this directory as MINICORE_DATA to your ~/.bashrc? y/n" yn
   case $yn in
        [Yy]* ) printf "#Setting MINICORE_DATA, for CSR-format single-cell data.\nMINICORE_DATA=$PWD\n" >> ~/.bashrc; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
export MINICORE_DATA=$PWD
