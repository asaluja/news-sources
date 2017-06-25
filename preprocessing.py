#!/usr/bin/env python

import argparse
import collections
import csv
import io
import re

import numpy as np
import pandas as pd

SUBJ_THRESHOLD = 100000000

def process_textfile(filehandle):
	records = []
	for line in filehandle:
		elements = line.strip().split()
		try:
			article_id = int(elements[0][2:])
		except ValueError:
			continue
		except IndexError:
			continue
		text = ' '.join(elements[1:])
		records.append((article_id, text))
	cols = ['article_id', 'text']
	df = pd.DataFrame.from_records(records, columns=cols)
	return df

# here: modify to handle multiple columns for source names
def old_parse_sources_file(filehandle):	
	output_files = {}
	for line in filehandle:
		source, filename = line.strip().split(':')[:2]
		output_files[source] = filename
	return output_files

def parse_sources_file(filename):	
	ns_dict = {}
	with io.open(filename, 'rU') as tsv:
		reader = csv.reader(tsv, dialect='excel-tab')
		header = reader.next()
		for line in reader:
			clean_line = filter(None, line)			
			assert len(clean_line) > 1
			ns_dict[clean_line[0]] = set(clean_line[1:])	
	return ns_dict


def replace_entities(text, entities):
	order = 2 # hardcoding to bigrams	
	text_vec = text.split()
	ngram_tuples = zip(*[text_vec[i:] for i in range(order)])
	ngram_vec = [' '.join(ngram_tuple) for ngram_tuple in ngram_tuples]
	intersection = entities.intersection(ngram_vec)
	if len(intersection) > 0:
		for entity in intersection:
			unigram_entity = '_'.join(entity.split())
			text = re.sub(entity, unigram_entity, text)
	return text

def clean_text(text, entities):
	text = text.lower()
	text = re.sub(r'<.*?>', '', text)
	text = re.sub('@ @ @ @ @ @ @ @ @ @', '<redacted>', text)
	text = replace_entities(text, entities)
	text = text.strip()
	return text

def compute_participant_histograms(filename, news_sources, word_counts):	
	all_ratings = pd.read_csv(filename, sep='\t').to_dict(orient='records')	
	subj_word_distr = {}
	subj_avg_rating_distr = {}
	for subj_rating in all_ratings:
		distr = np.zeros(7)
		for source in news_sources:
			rating = int(subj_rating[source])
			distr[rating + 3] += clean_word_counts[source]
		avg_rating = np.divide(np.dot(range(7) - 3.*np.ones(7), distr), np.sum(distr))
		subj_avg_rating_distr[subj_rating['Subj']] = avg_rating
		subj_word_distr[subj_rating['Subj']] = distr	
	#vals = [[], [], []]
	#print(subj_avg_rating_distr)
	#for subject in subj_word_distr:				
	#	temp = list(reversed([wc for wc in subj_word_distr[subject] if wc > 0]))[:3]
	#	if len(temp) == 1:
	#		continue		
	#	for counter, word_count in enumerate(temp[:3]):
	#		vals[counter].append(word_count)
	vals = []				
	for subject in subj_word_distr:
		min_index = int(np.ceil(subj_avg_rating_distr[subject]))
		temp = subj_word_distr[subject][min_index:]
		vals.append(np.sum(temp))

	#vals_arr = np.array([np.array(xi) for xi in vals]).T	
	#print(np.divide(np.mean(vals_arr, axis=0), np.sum(np.mean(vals_arr, axis=0))))	
	#temp_arr = np.sum(vals_arr, axis=1)
	temp_arr = np.array([np.array(xi) for xi in vals]).T	
	print(np.mean(temp_arr))
	print(np.std(temp_arr))	
	hist, bin_edges = np.histogram(temp_arr)
	print(hist)
	print(bin_edges)
	import matplotlib.pyplot as plt
	plt.hist(hist, bins=bin_edges)
	plt.show()


def sample_news_sources(news_sources, word_counts, words_to_sample):
	words_sampled = 0
	sampled_sources = []
	while words_sampled < words_to_sample:
		sampled_source = np.random.choice(news_sources, replace=False)
		words_sampled += word_counts[sampled_source]
		sampled_sources.append(sampled_source)
	return sampled_sources, words_sampled


def select_corpora(subj_filename, word_counts):
	all_ratings = pd.read_csv(subj_filename, sep='\t').to_dict(orient='records')	
	participant_corpora = {}
	for idx, subj_rating in enumerate(all_ratings): # here: dict of news source --> rating				
		acc_sources = []
		ratings_by_group = []		
		for desired_rating in range(1, 4):
			ratings_by_group.append([ns for ns, rating in subj_rating.items() if rating == desired_rating])			
		total_words = 0
		for news_sources in reversed(ratings_by_group):
			if news_sources:
				num_words = sum(word_counts.get(source, 0) for source in news_sources)
				if total_words + num_words < SUBJ_THRESHOLD:
					acc_sources.extend(news_sources)
					total_words += num_words
				else: # then we need to sample
					sampled_sources, sampled_words = sample_news_sources(news_sources, word_counts, SUBJ_THRESHOLD - total_words)
					acc_sources.extend(sampled_sources)
					total_words += sampled_words
					break
			else:
				continue
		if total_words < SUBJ_THRESHOLD:
			print("Number of words for subject %d above rating 0: %d; rejecting..." % (idx, total_words))
		else:
			participant_corpora[idx] = (acc_sources, total_words)
	return participant_corpora

def output_participant_corpora(output_dir, text_df, metadata_df, entities, subject_corpora):
	for idx, corpora_num_words in subject_corpora.items():
		if idx != 15:
			continue
		corpora, num_words = corpora_num_words
		out_filename = output_dir + '/%d.txt' % idx
		with io.open(out_filename, 'wb') as output_fh:
			filtered_metadata = metadata_df[metadata_df['source'].isin(corpora)]			
			source_text = filtered_metadata.merge(text_df, on='article_id', suffixes=('_meta', '_text'))['text']
			source_text_clean = source_text.apply(lambda x: clean_text(x, entities))	
			for article in source_text_clean.values:
				output_fh.write("%s\n" % article)
			print("Subject %d: wrote %d words across %d articles to %s" % (idx, num_words, len(source_text), out_filename))	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--metadata", help="TSV containing artilce ID and other info.")	
	parser.add_argument("--metadata_out", action="store", default="")
	parser.add_argument("--news_sources", help="text file containing name of news source and corresponding output filename, separated by ':' e.g., New York Times: nyt_out.txt")	
	parser.add_argument("--entities", help="predefined entities")
	parser.add_argument("--subjects", action="store")	
	parser.add_argument("--text", help="file containing article IDs and text")
	parser.add_argument("--output")
	args = parser.parse_args()

	# process metadata
	metadata_df = pd.read_csv(args.metadata, sep='\t', header=None, names=['article_id', 'num_words', 'ds', 'country', 'source', 'url', 'title'])	
	metadata_df.drop_duplicates(subset=['article_id'], inplace=True)
	metadata_df['country'] = metadata_df['country'].str.strip()
	metadata_df['source'] = metadata_df['source'].str.strip()
	metadata_df['title'] = metadata_df['title'].str.strip()	
	print("Total number of words across %d documents: %d" % (len(metadata_df), metadata_df['num_words'].sum()))	
	#words_by_source = metadata_df.groupby('source').agg({'num_words': [np.size, np.sum]}).sort_values(by=[('num_words', 'sum')], ascending=False)		

	#process news sources
	news_sources = parse_sources_file(args.news_sources)	
	all_pubs = set.union(*news_sources.values())
	print("Out of %d unique news sources, we have %d sources to look for" % (len(news_sources), len(all_pubs)))		
	filtered_metadata_df = metadata_df[metadata_df['source'].isin(all_pubs)]
	print("Using %d articles out of total." % (len(filtered_metadata_df)))	
	print("Total number of words in %d unique news sources: %d" % (len(news_sources), filtered_metadata_df['num_words'].sum()))
	if args.metadata_out:
		words_by_source = filtered_metadata_df.groupby('source').agg({'num_words': [np.size, np.sum]}).sort_values(by=[('num_words', 'sum')], ascending=False)
		words_by_source.to_csv(args.metadata_out, sep='\t')
	word_counts_list = zip(filtered_metadata_df['source'].values, map(int, filtered_metadata_df['num_words'].values))
	word_counts = collections.defaultdict(int)
	for source, count in word_counts_list:
		word_counts[source] += count	
	clean_word_counts = {}
	for source_clean in news_sources:
		clean_word_counts[source_clean] = sum(word_counts[source] for source in news_sources[source_clean])	
	print("Metadata and news source processing complete.")
	entities_fh = io.open(args.entities, 'rb')
	entities = set([line.strip().lower() for line in entities_fh.readlines()])
	entities_fh.close()	
	print("Finished reading and processing news sources and entities files.")
		
	#compute_participant_histograms(args.subjects, news_sources, word_counts)
	participant_corpora = select_corpora(args.subjects, clean_word_counts)				
	text_fh = io.open(args.text, 'rb')
	text_df = process_textfile(text_fh)
	text_fh.close()	
	print("Finished reading in text file.")
	filtered_text_df = filtered_metadata_df.merge(text_df, on='article_id', suffixes=('_meta', '_text'))
	filtered_text_df['text_clean'] = filtered_text_df['text'].apply(lambda x: clean_text(x, entities))			
	with io.open(args.output + '/all_text.txt', 'wb') as output_fh:		
		for article in filtered_text_df['text_clean'].values:
			output_fh.write("%s\n" % article)
	print("Finished writing all text.")		
	output_participant_corpora(args.output, filtered_text_df, filtered_metadata_df, entities, participant_corpora)	
	

if __name__ == "__main__":
	main()