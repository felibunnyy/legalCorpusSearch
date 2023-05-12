#!/usr/bin/python3
import re
import nltk
import sys
import getopt

from postlist import *
import pickle

from collections import OrderedDict

from nltk.tokenize import word_tokenize
from nltk import stem
from nltk.corpus import stopwords

import importlib
import pip

sys.setrecursionlimit(50000)

# python3 index.py -i dataset.csv -d dictionary_file -p postings_file
def install_package(package):
    try:
        importlib.import_module(package)
    except ImportError:
        pip.main(['install', package])

install_package('pandas')

import pandas as pd

stemmer = stem.PorterStemmer()

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

def get_term(text):
    """
    Pre-process text (Tokenize, Stemming, Case Holding)
    """
    result = []
    tokens = word_tokenize(text)
    for token in tokens:
        term = stemmer.stem(token)
        term = term.lower()
        result.append(term)
    return result

def update_index(index, docID, tokens, eachDocId_length_for_normalisation):
    """
    Add token to index, calculate length of doc
    """

    positional_index = 1

    for token in tokens:
        if(token not in index):
            # Add term to index
            plist = PostingList()
            plist.insertNode(docID, positional_index) #first positional index
            index[token] = plist
        else:
            # Update index
            index[token].insertNode(docID, positional_index)
        
        positional_index += 1
    
        if (token not in eachDocId_length_for_normalisation):
            eachDocId_length_for_normalisation[token] = 1
        else:
            eachDocId_length_for_normalisation[token] += 1

def calculate_normalised_length(eachDocId_length_for_normalisation):
    temp = 0
    for key, value in eachDocId_length_for_normalisation.items():
        temp += (1 + math.log10(value))**2

    return math.sqrt(temp)

def build_index(in_file, out_dict, out_postings):
    """
    build index from corpus given in the input file,
    then output the dictionary file and postings file
    """
    print('indexing...')

    data = pd.read_csv(in_file)
    data = data.sort_values('document_id')
    
    # print(data)

    index = {}
    
    final_calculated_normalised_length = {}
    full_doc_ids = []
    
    for idx, row in data.iterrows():
        eachDocId_length_for_normalisation = {}
        docID = row['document_id']
        full_doc_ids.append(docID)

        #preprocess text:
        text_to_process = row['content']
        text_to_process = text_to_process.replace('\n', ' ')
        text_to_process = ''.join(char for char in text_to_process if char.isalnum() or char.isspace())
        
        tokens = get_term(text_to_process)
        update_index(index, docID, tokens, eachDocId_length_for_normalisation)

        # calculate final_calculated_normalised_length for current docId
        final_calculated_normalised_length[docID] = calculate_normalised_length(eachDocId_length_for_normalisation)
    
    print('finished indexing')

    postlist_file = open(out_postings, 'wb')
    seek_value_count = 0

    output_dict = {} # dictionary with mapping of word to [df, pickle seek value count]

    index = OrderedDict(sorted(index.items()))
    for key, value in index.items():
        temp_item_posting_list = value
        output_dict[key] = [temp_item_posting_list.df, seek_value_count]
        pickle.dump(temp_item_posting_list, postlist_file, protocol = 4)
        seek_value_count = postlist_file.tell()

    print('finished dumping post file')

    full_doc_ids = tuple(full_doc_ids)


    # postlist_file contains posting list dumped
    with open(out_dict, "wb") as index_file:
        pickle.dump(output_dict, index_file, protocol = 4)
        pickle.dump(full_doc_ids, index_file, protocol = 4)
        pickle.dump(final_calculated_normalised_length, index_file, protocol = 4)
    
    print('done')

input_file = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input file
        input_file = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_file == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_file, output_file_dictionary, output_file_postings)
