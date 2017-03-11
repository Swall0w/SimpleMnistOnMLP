# SimpleMnistOnMLP
The max size of file volume is limited to 100MB in github.

So, plz concat 'mnist_train.csv.*' files. to generate a complete file.

## Split file
    $ split -b 50000000 test.csv test.csv.

## Concat file
    $ cat test.csv.* > test.csv


