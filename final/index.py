#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import math

import importlib

def install_package(package):
    try:
        importlib.import_module(package)
    except ImportError:
        pip.main(['install', package])

install_package('pandas')
install_package('nltk')

import pickle
import time

from collections import OrderedDict, Counter

from nltk.tokenize import word_tokenize
from nltk import stem
from nltk.corpus import stopwords

import os
import shutil

stop_words = stopwords.words('english')

import pip

sys.setrecursionlimit(50000)

import pandas as pd

stemmer = stem.PorterStemmer()

##ZONES AND FIELDS
# note: court name processing is only 'to_lower'
courts_most_impt = ["SG Court of Appeal", "SG Privy Council", "UK House of Lords", "UK Supreme Court",
                "High Court of Australia", "CA Supreme Court"]

courts_less_impt = ["SG High Court", "Singapore International Commercial Court", "HK High Court",
                "HK Court of First Instance", "UK Crown Court", "UK Court of Appeal", "UK High Court", "Federal Court of Australia",
                "NSW Court of Appeal", "NSW Court of Criminal Appeal", "NSW Supreme Court"] #but still more important than those not even listed

def encode_varbyte(n):
    bytes_needed = math.ceil(n.bit_length() / 7)
    result = bytearray()
    for i in range(bytes_needed):
        byte = n & 0x7F
        n >>= 7
        if i != bytes_needed - 1:
            byte |= 0x80
        result.append(byte)
    return bytes(result)

def decode_varbyte(encoded):
    value = 0
    shift = 0
    for byte in encoded:
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            break
        shift += 7
    return value

def compress_index(index):
  for key, value in index.items():
    new_value = []
    for each_tuple in value:
        positional_indexes = each_tuple[2]
        counter_position = positional_indexes[0]
        encoded_array = [encode_varbyte(positional_indexes[0])]
        for i in range(1, len(positional_indexes)):
            real_position = positional_indexes[i]
            offset = real_position - counter_position
            counter_position = real_position
            encoded_array.append(encode_varbyte(offset))
        encoded_tuple = tuple(encoded_array)
        new_each_tuple = (each_tuple[0], each_tuple[1], encoded_tuple)
        new_value.append(new_each_tuple)
    new_value = tuple(new_value)
    index[key] = new_value
      
# python3 index2.py -i dataset.csv -d dictionary_file4 -p postings_file4

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

def get_term(text):
    """
    Pre-process text (Tokenize, Stemming, Case Holding)
    """
    result = []
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in stop_words]
    for token in tokens:
        result.append(stemmer.stem(token).lower())
    return result

def preprocess_text(text_to_process):
    text_to_process = text_to_process.replace('\n', ' ')
    text_to_process = ''.join(char for char in text_to_process if char.isalnum() or char.isspace())
    return text_to_process

def update_index(index, docID, tokens):
    """
    Add token to index, calculate length of doc
    """

    positional_index = 1

    for token in tokens:
      init_tf = 1
      if (token not in index):
          # Add term to index
          index[token] = [(docID, init_tf, [positional_index])]
      else:
          # Update index
          last_seen_docId = index[token][-1][0]
          if (last_seen_docId == docID): #docId already exists in posting list
            index[token][-1][2].append(positional_index)
            index[token][-1] = (index[token][-1][0], index[token][-1][1] + 1, index[token][-1][2])
            # postlist = index[token][-1]
            # postlist[1] += 1 # update tf
            # postlist[2].append(positional_index)
          else: #docId is new in posting list
            index[token].append((docID, init_tf, [positional_index]))

      positional_index += 1

def update_doc_vector(doc_vector, docID, terms):
    if docID not in doc_vector:
        doc_vector[docID] = Counter(terms)
    else:
        doc_vector[docID] += Counter(terms)

def make_dir(dir_name):
    if(os.path.exists(dir_name)):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def create_blocks_index(data, court_name_to_docId_mapping, zones_and_fields_dict, doc_vector, threshold_term):
    make_dir('blockPath')
    num_block_files_count = 0
    while (len(data) > 0):
        num_block_files_count += 1
        #print('block files = {}'.format(num_block_files_count))
        block_dictionary = {}
        while (len(block_dictionary) < threshold_term and len(data) > 0):
            docID = data['document_id'].iloc[0]
            title = data['title'].iloc[0]
            court = data['court'].iloc[0]
            date_posted =  data['date_posted'].iloc[0]
            extracted_date = date_posted[:10]
            
            ###court_name_to_docId_mapping
            if court in courts_less_impt or court in courts_most_impt:
                if court.lower() not in court_name_to_docId_mapping:
                    court_name_to_docId_mapping[court.lower()] = [docID]
                else:
                    court_name_to_docId_mapping[court.lower()].append(docID)
            ###
            
            processed_title = preprocess_text(title)
            processed_title = get_term(processed_title)
            title_array = [] #not keeping track of tf, but unique words that appear
            for word in processed_title:
                if word not in title_array:
                    title_array.append(word)
            title_tuple = tuple(title_array)
            zones_and_fields_dict[docID] = (title_tuple, court.lower(), extracted_date)
 
            #preprocess text:
            text_to_process = data['content'].iloc[0]
            text_to_process = preprocess_text(text_to_process) #processing done
        
            tokens = get_term(text_to_process)
            update_index(block_dictionary, docID, tokens)
            update_doc_vector(doc_vector, docID, tokens)

            tokens = get_term(data['content'].iloc[0])
            update_index(block_dictionary, docID, tokens)
        
            #print('block dict = {}'.format(len(block_dictionary)))
 
            data.drop(data.index[0], inplace=True)
            
        #after exiting inner while loop, one block_dictionary is done
        block_dictionary = OrderedDict(sorted(block_dictionary.items()))
        print('rows left = {}'.format(len(data)))
        print('block dict = {}'.format(len(block_dictionary)))
            
        #open new file connection and store block_dictionary with id = num_block_files_count
        temp_block_dict = 'blockPath' + '/' + str(num_block_files_count) + '.pickle'
        with open(temp_block_dict, "ab") as f:
            for item in block_dictionary.items():
                pickle.dump(item, f)
    
    return num_block_files_count

def merge_postlist(first, second):
    print(first, second)
    counter1, counter2 = 0, 0
    result = []
    while counter1 < len(first) or counter2 < len(second):
        if first[counter1][0] == second[counter2][0]:
            positions = first[counter1][2] + second[counter2][2]
            positions.sort()
            result.append((first[counter1][0], first[counter1][1] + second[counter2][1], positions))
            counter1 += 1
            counter2 += 1
        elif first[counter1][0] < second[counter2][0]:
            result.append((first[counter1][0], first[counter1][1], first[counter1][2]))
            counter1 += 1
        elif first[counter1][0] > second[counter2][0]:
            result.append((second[counter1][0], second[counter1][1], second[counter1][2]))
            counter2 += 1
    return result

def merge_index(num_block_files_count):
    ### 2. Perform merging bit by bit: Writing the confirmed entries (known by cut_off_temp) into a pickled object in file (1 for every iteration).
    ### Each such file is stored under 'temp' directory
    #perform merging bit by bit in memory and construction of output_file_dictionary
    make_dir('temp')
    temp_path_file_count = 0
    temp_loaded_dict_in_memory = {}
    cut_off_temp = False
    total_num_blocks = num_block_files_count #starting from index 1
    flag_checked_at_least_one_doc = True
    tell_values = [0] * (total_num_blocks + 1) #contains tell values so we can seek in the next iteration
    while (flag_checked_at_least_one_doc):
        temp_path_file_count += 1
        flag_checked_at_least_one_doc = False
        for block_num in range (1, total_num_blocks + 1):
            curr_iter_block_path = 'blockPath' + '/' + str(block_num) + '.pickle'
            with open (curr_iter_block_path, "rb") as pickled_file:
                try:
                    start = tell_values[block_num]
                    pickled_file.seek(start)
                    for i in range(1000):
                        temp_retrieval = pickle.load(pickled_file) #(term, posting list)
                        if (i == 0):
                            flag_checked_at_least_one_doc = True
                        temp_key = temp_retrieval[0]
                        if (i == 0 and cut_off_temp == False):
                                cut_off_temp = temp_key
                        else:
                            if (temp_key < cut_off_temp):
                                cut_off_temp = temp_key
                        temp_value_array = temp_retrieval[1]
                        if (temp_key in temp_loaded_dict_in_memory):
                            #start from the first index, if any
                            temp_loaded_dict_in_memory[temp_key] = merge_postlist(temp_value_array, temp_loaded_dict_in_memory[temp_key]) 
                        else: #new key in temp_loaded_dict_in_memory
                            temp_loaded_dict_in_memory[temp_key] = temp_value_array   
                    tell_values[block_num] = pickled_file.tell()
                    pickled_file.close()
                except EOFError:
                    tell_values[block_num] = pickled_file.tell()
                    pickled_file.close()
                    
        #write in-memory dictionary into hard disk until cut_off is found:
        temp_loaded_dict_in_memory = OrderedDict(sorted(temp_loaded_dict_in_memory.items()))
        temp_output_file_postings = 'temp/' + str(temp_path_file_count)
        f = open(temp_output_file_postings, "wb")
        if (cut_off_temp == False): #wont go into next iteration of while loop
            if (len(temp_loaded_dict_in_memory) > 0):
                # for item in temp_loaded_dict_in_memory.items():
                pickle.dump(temp_loaded_dict_in_memory, f)
        else:
            if (len(temp_loaded_dict_in_memory) == 0):
                #checked jic
                cut_off_temp = False
                continue
            else:
                dumping_dict = {} #created so construction of in memory dictionary will be easier, dunnid pickle.load each entry once and can dump away easier too
                for key, value in temp_loaded_dict_in_memory.items():
                    if (key == cut_off_temp): #check if curr key == cut_off_temp
                        dumping_dict[key] = value
                        break
                    else:
                        dumping_dict[key] = value
                #re-ensure dumping_dict is sorted
                dumping_dict = OrderedDict(sorted(dumping_dict.items()))
                pickle.dump(dumping_dict, f)
                last_key = list(temp_loaded_dict_in_memory.keys())[-1]
                #resetting temp_loaded_dict_in_memory for next iteration
                if (last_key == cut_off_temp):
                    #reset temp_loaded_dict_in_memory to empty
                    temp_loaded_dict_in_memory = {}
                else:
                    #keep only the terms after cut_off_temp onwards
                    for k in list(temp_loaded_dict_in_memory.keys()):
                        if (k != cut_off_temp):
                            del temp_loaded_dict_in_memory[k]
                        else: # k == cut_off_temp
                            #delete cut_off_temp now and break out of for loop
                            del temp_loaded_dict_in_memory[cut_off_temp]
                            break
                    
        f.close()
        
        #reset value for next iteration
        cut_off_temp = False

def output_final_index(output_file_dictionary):
    ### 3. Now create the in_memory_dictionary that is eventually stored in 'output_file_dictionary'. in_memory_dict = {"term1":(docfreq1, seekvalue1), "term2":(docfreq2, seekvalue2), etc.}
    ### Simultaneously write final posting list one by one (as pickled object) into 'output_file_postings'
    
    in_memory_dictionary = {}
    
    #load (now sorted) pickled files from 'temp' directory part by part to construct in_memory_dictionary
    
    seek_value_count = 0
    ordered_hard_disk_dictionary_path_files = sorted(map(int, os.listdir('temp')))
    
    f = open(output_file_postings, "ab") #simultaneously write final posting list into output_file_postings entry by entry
    
    for filename in ordered_hard_disk_dictionary_path_files:
        file_path = 'temp/' + str(filename)
        with open (file_path, "rb") as pickled_file:
            try:
                temp_dict_retrieval = pickle.load(pickled_file)
                for key, value in temp_dict_retrieval.items():
                    if(value is not None):
                        temp_key = key
                        in_memory_dictionary[temp_key] = [len(value), seek_value_count]
                        pickle.dump(value, f)
                        seek_value_count = f.tell()
            except EOFError:
                    pass

    #write in_memory_dictionary to output_file_dictionary to load into search.py
     
    with open(output_file_dictionary, "wb") as f:
        pickle.dump(in_memory_dictionary, f)

def build_index(in_file, out_dict, out_postings):
    """
    build index from corpus given in the input file,
    then output the dictionary file and postings file
    """
    print('indexing...')
    startTime = time.time()
    
    data = pd.read_csv(in_file)
    # data = data.sort_values('document_id')

    data = data.head(100)

    # rows_left = len(data)
    
    index = {}
    doc_vector = {}

    # number of unique docIDs     
    N = len(data['document_id'].unique())
        
    #court_mappings = score of 2 if it is under courts_most_impt, score of 1 if it is under courts_less_impt, score of 0 otherwise
    zones_and_fields_dict = {} #maps docId to tuple of 1. dict of word in title to tf
                                                       #2. court name (lowercase)
                                                       #3. date posted tuple of 3 values (yyyy, mm, dd)
    court_name_to_docId_mapping = {} #maps court name to docId
    court_mapping = {} # maps name of court to a special id
    court_score_mapping = {} # maps special id to court importance score
    court_counter_for_mapping = 1
    for court in courts_most_impt:
        court = court.lower()
        court_mapping[court] = court_counter_for_mapping
        court_score_mapping[court_counter_for_mapping] = 2
        court_counter_for_mapping += 1
    for court in courts_less_impt:
        court = court.lower()
        court_mapping[court] = court_counter_for_mapping
        court_score_mapping[court_counter_for_mapping] = 1
        court_counter_for_mapping += 1
    
    num_block_files_count = create_blocks_index(data, court_name_to_docId_mapping, zones_and_fields_dict, doc_vector, 10000)
    merge_index(num_block_files_count)


    # for idx, row in data.iterrows():
    #     docID = row['document_id']
    #     title = row['title']
    #     court = row['court']
    #     date_posted = row['date_posted']
    #     extracted_date = date_posted[:10]
        
    #     ###court_name_to_docId_mapping
    #     if court in courts_less_impt or court in courts_most_impt:
    #         if court.lower() not in court_name_to_docId_mapping:
    #             court_name_to_docId_mapping[court.lower()] = [docID]
    #         else:
    #             court_name_to_docId_mapping[court.lower()].append(docID)
    #     ###
        
    #     processed_title = preprocess_text(title)
    #     processed_title = get_term(processed_title)
    #     title_array = [] #not keeping track of tf, but unique words that appear
    #     for word in processed_title:
    #         if word not in title_array:
    #             title_array.append(word)
    #     title_tuple = tuple(title_array)
    #     zones_and_fields_dict[docID] = (title_tuple, court.lower(), extracted_date)
 
    #     #preprocess text:
    #     text_to_process = row['content']
    #     text_to_process = preprocess_text(text_to_process) #processing done
        
    #     tokens = get_term(text_to_process)
    #     update_index(index, docID, tokens)
    #     update_doc_vector(doc_vector, docID, tokens)

    #     # progress checking
    #     # rows_left -= 1
    #     print('finished processing docID = {}'.format(docID))

    # calculate_weight(index)

    # index = OrderedDict(sorted(index.items()))
    compress_index(index)
    
    print('finished indexing')

    postlist_file = open(out_postings, 'wb')
    seek_value_count = 0

    output_dict = {} # dictionary with mapping of word to [df, pickle seek value count]

    for term, postlist in index.items():
        output_dict[term] = [len(postlist), seek_value_count]
        pickle.dump(postlist, postlist_file, protocol = 4)
        seek_value_count = postlist_file.tell()

    print('finished dumping post file')

    # postlist_file contains posting list dumped
    with open(out_dict, "wb") as index_file:
        pickle.dump(output_dict, index_file, protocol = 4)
        pickle.dump(N, index_file, protocol = 4)
        pickle.dump(zones_and_fields_dict, index_file, protocol = 4)
        pickle.dump(court_mapping, index_file, protocol = 4)
        pickle.dump(court_score_mapping, index_file, protocol = 4)

    with open('docvec.txt', "wb") as docvec_file:
        pickle.dump(doc_vector, docvec_file, protocol=4)
    
    print('done')
    print ("Execution Time:" + str(time.time() - startTime) + "s")

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