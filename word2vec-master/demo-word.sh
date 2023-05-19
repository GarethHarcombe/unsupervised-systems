make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time ./word2vec -train text8 -output vectors1.bin -cbow 0 -size 50 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 10
time ./word2vec -train text8 -output vectors2.bin -cbow 0 -size 50 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 10
time ./word2vec -train text8 -output vectors3.bin -cbow 0 -size 50 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 10
time ./word2vec -train text8 -output vectors4.bin -cbow 0 -size 50 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 10
time ./word2vec -train text8 -output vectors5.bin -cbow 0 -size 50 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 10
./distance vectors.bin
