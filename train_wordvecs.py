#!/usr/bin/env python

import argparse
from gensim.models import word2vec

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='baseline', help="training mode.")
	parser.add_argument('--pretrained', default='')
	parser.add_argument('--corpus', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--print_vectors', action='store_true')
	args = parser.parse_args()

	model = word2vec.Word2Vec(size=100, window=5, min_count=1, workers=4, sg=1)
	sentences = word2vec.LineSentence(args.corpus)
	if args.mode == 'participant': # load word vectors here, then proceed with training
		assert args.pretrained		
		model = word2vec.Word2Vec.load(args.pretrained)
	else: # build vocab
		model.build_vocab(sentences)		
	model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)		
	# save model here
	model.save(args.output)
	if args.print_vectors:
		for word in model.wv.vocab.keys():
			print("%s\t%s" % (word, ' '.join(map(str, model.wv[word]))))


if __name__ == '__main__':
	main()