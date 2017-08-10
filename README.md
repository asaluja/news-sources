# news-sources

## Preprocessing:

Assuming this directory is called `code/` and the data is in `data/`:

```
python code/preprocessing.py --metadata data/all_sources.txt --text data/all_text.txt --news_sources data/news_sources.txt --entities data/entities.txt --subjects data/participants_v2.txt --output experiments/
```

This script generates participant-specific corpora (all text in one file per participant) in the directory specified by `output`, and also writes out a file `all_text.txt` in the same directory.  The `all_text.txt` file contains text from all of the news sources together. 

Regarding the inputs:
- `all_sources.txt`: metadata containing aritcle ID, number of words in article, date released, country, news source, and URL.
- `all_text.txt`: one file containing the entire NOW corpus. 
- `news_sources.txt`: file that you gave me that maps the variations in news source names to a unique identifier. 
- `entities.txt`: the entities you care about, so that I represent them as single words (i.e., "Donald Trump" --> "donald_trump") in the corpus
- `participants_v2.txt`: participant ratings (v2 because it's the larger file you sent). 

## To run word vector training:

```
python code/train_wordvecs.py --corpus experiments/all_text.txt --mode baseline --output all_text.vecs

```

This first creates the baseline word vectors. 

Then:
```
for filename in experiments/*.txt; do python code/train_wordvecs.py --corpus $filename --mode participant --pretrained all_text.vecs --output $filename.vecs --print_vectors > $filename.out; done
```
