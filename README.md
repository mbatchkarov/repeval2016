### Description
This repository contains the code to reproduce the results of the 2016 paper "A critique of word similarity as a method for evaluating distributional semantic models" by Batchkarov, Kober, Reffin, Weeds and Weir.

### Usage

 -  Get some text data. Decompressing it is not necessary. Change the bit that says `open(join(self.dirname, fname))` to `gzip.open(join(self.dirname, fname))`.

```
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
unzip text8.gz
```

 -  Install required Python dependencies (Py3, pandas, numpy).

 -  Train a `word2vec` model

```
python train_word2vec --input-dir raw_text --output-file vectors/wtv.gs
```

 -  Generate random vectors as a baseline

```
python generate_random_vectors.py
```

 -  Run evaluation

```
python intrinsic_eval.py
```

 -  Inspect results using Jupyter Notebooks