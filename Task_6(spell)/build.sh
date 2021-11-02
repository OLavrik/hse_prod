
python3 -m spacy download en_core_web_sm
mkdir -p "./resource"
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.dic -P "./resource"
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.aff -P "./resource"
wget http://aspell.net/test/cur/batch0.tab -P "./resource"
wget https://www.norvig.com/ngrams/spell-errors.txt -P "./resource"
wget https://zenodo.org/record/1199620/files/SO_vectors_200.bin?download=1 "."

