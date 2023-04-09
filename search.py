#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import os
import pickle

from nltk.tokenize import word_tokenize
from nltk import stem
import math

from postlist import *

stemmer = stem.PorterStemmer()

# python3 search.py -d dictionary-file -p postings-file -q queries_file.txt -o output-file-of-results

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

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

def phrase_helper(first, second, in_memory_dictionary, posting_list_file): #returns a pl of relevant docs
    temp_df1, temp_pickled_index1 = in_memory_dictionary[first]
    posting_list_file.seek(temp_pickled_index1)
    temp_pl1 = pickle.load(posting_list_file)
    temp_df2, temp_pickled_index2 = in_memory_dictionary[second]
    posting_list_file.seek(temp_pickled_index2)
    temp_pl2 = pickle.load(posting_list_file)
    common_docIds = and_query(temp_pl1, temp_pl2, in_memory_dictionary, posting_list_file)
    
    #traverse through common_docIds posting list
    currNode = common_docIds.head
    currNode1 = temp_pl1.head
    currNode2 = temp_pl2.head
    
    relevant_results = [] #should we create another posting list with just the docId
    while (currNode is not None):
        docId = currNode.data
        while (currNode1.data != docId):
            currNode1 = currNode1.next
        while (currNode2.data != docId):
            currNode2 = currNode2.next
        if (currNode1.data == docId and currNode2.data == docId):
            positionalIndexes1 = currNode1.positions
            positionalIndexes2 = currNode2.positions
            j = 0
            for i in range(len(positionalIndexes1)):
                temp_index = positionalIndexes1[i]
                while (positionalIndexes2[j] <= temp_index):
                    j += 1
                if (positionalIndexes2[j] == positionalIndexes1[i] + 1):
                    relevant_results.append(docId)
                    break
        else:
            raise KeyError
    
    return relevant_results
    
#if overall query is a boolean query
def phrase_query(phrase_query, in_memory_dictionary, posting_list_file):
    tokens = get_term(phrase_query)
    relevant_docs = PostingList()
    for token in tokens: #preliminary check
        if (token not in in_memory_dictionary): #word: [df, pickle index]
            return relevant_docs
    
    
    if (len(phrase_query) == 2):
        resulting_pl = phrase_helper(token[0], token[1], in_memory_dictionary, posting_list_file)
        return resulting_pl
    
    elif (len(phrase_query) == 3):
        temp_pl1 = phrase_helper(token[0], token[1], in_memory_dictionary, posting_list_file)
        temp_pl2 = phrase_helper(token[1], token[2], in_memory_dictionary, posting_list_file)
        return and_query(temp_pl1, temp_pl2, in_memory_dictionary, posting_list_file)
    else:
        raise TypeError("length of phrase query should be 2 or 3")
        
    

def find_results(query, in_memory_dictionary, posting_list_file):
    tokens = get_term(query)
    cosine_without_normalisation = {} # docId: second vector, where second vector is a dict {mappingid: 1 + logtf term query value}
    curr_query_term_freq = {} # maps mappingid to [term freq in entire query, df in index] (for first vector - query)
    curr_query_term_freq_mapping = {} # maps term to number, where number = mappingid

    current_counter = 0
    
    for token in tokens:
        if(token in in_memory_dictionary):
            seek_val = in_memory_dictionary[token][1]
            posting_list_file.seek(seek_val)
            posting_list = pickle.load(posting_list_file)

            if (token in curr_query_term_freq_mapping):
                mappingid = curr_query_term_freq_mapping[token]
                curr_query_term_freq[mappingid][0] += 1
            elif (token not in curr_query_term_freq_mapping):
                query_term_df = posting_list.df
                curr_query_term_freq_mapping[token] = current_counter
                mappingid = current_counter
                current_counter += 1
                curr_query_term_freq[mappingid] = [1, query_term_df]
            
                # calculation of first (query) vector in dot product(for 'cosine_without_normalisation')
                curr = posting_list.head

                while(curr is not None):
                    currDocId = curr.data
                    if (currDocId not in cosine_without_normalisation):
                        cosine_without_normalisation[currDocId] = {}
                    
                    cosine_without_normalisation[currDocId][mappingid] = 1 + math.log10(curr.tf)
                    curr = curr.next
    
    return curr_query_term_freq, cosine_without_normalisation

def calculate_tf_idf(N, curr_query_term_freq, cosine_without_normalisation, final_calculated_normalised_length):
    # processing curr_query_term_freq to calculate tf-idf for each unique term in query
    for key, value in curr_query_term_freq.items():
        tf = 1 + math.log10(value[0])
        idf = math.log10(N/value[1])
        tf_idf = tf * idf
        # update value of each key to tf_idf value
        curr_query_term_freq[key] = tf_idf

    # start calculating cosine similarity for each document
    normalised_query_length = 0
    for key, value in curr_query_term_freq.items():
        normalised_query_length += value**2
    normalised_query_length = math.sqrt(normalised_query_length)

    for key, value in cosine_without_normalisation.items():
        counter = 0
        for key2, value2 in value.items(): # doing dot product before normalising
            counter += curr_query_term_freq[key2] * value2 # query val * doc val for common terms between query and doc
        
        # divide by the normalising length in 'final_calculated_normalised_length' (for doc vec length)
        counter /= final_calculated_normalised_length[key] # key = docId
        
        # divide by the normalising length, normalised_query_length (for query vec length)
        counter /= normalised_query_length
        
        # take normalised counter value and replace it with value in 'cosine_without_normalisation'
        cosine_without_normalisation[key] = counter
    
    sorted_result = sorted(cosine_without_normalisation.items(), key=lambda x:x[1])

    result = PostingList()
    for item in sorted_result:
        result.insertNode(item[0])

    return result

def and_query(first, second, in_memory_dictionary, p):
    if (type(first) != PostingList):
        seek_value_first = in_memory_dictionary[first][1]
        p.seek(seek_value_first)
        first = pickle.load(p)
    if (type(second) != PostingList):
        seek_value_second = in_memory_dictionary[second][1]
        p.seek(seek_value_second)
        second = pickle.load(p)

    # special cases:
    if (first.head == None or second.head == None):
        return PostingList()

    pointer1 = first.head
    pointer2 = second.head
    
    ans = PostingList()

    while (pointer1 is not None and pointer2 is not None):
        value1 = pointer1.data
        value2 = pointer2.data
        if (value2 < value1):
            if (pointer2.next is not None and pointer2.next.data <= value1):
                    pointer2 = pointer2.next
                    value2 = pointer2.data
            elif (pointer2.next == None): #already at last
                break
            else:
                pointer2 = pointer2.next
                value2 = pointer2.data
                
        elif (value1 < value2):
            if (pointer1.skip is not None and pointer1.skip.data <= value2):      
                    pointer1 = pointer1.skip
                    value1 = pointer1.data
            elif (pointer1.next == None): #already at last
                break
            else:
                pointer1 = pointer1.next
                value1 = pointer1.data
        
        elif (value2 == value1):
            ans.insertNode(value1)
            pointer1 = pointer1.next
            pointer2 = pointer2.next

    return ans

# KIV
def process_query(query, N, in_memory_dictionary, posting_list_file, final_calculated_normalised_length):
    queries = query.split(" AND ")

    if(len(queries) == 1):
        curr_query_term_freq, cosine_without_normalisation = find_results(query, in_memory_dictionary, posting_list_file)
        result = calculate_tf_idf(N, curr_query_term_freq, cosine_without_normalisation, final_calculated_normalised_length)
        print(result)
        return result
    
    else:
        result = []
        for q in queries:
            print(q)
            curr_query_term_freq, cosine_without_normalisation = find_results(query, in_memory_dictionary, posting_list_file)
            result.append(calculate_tf_idf(N, curr_query_term_freq, cosine_without_normalisation, final_calculated_normalised_length))
            print(result)

        while(len(result) > 1):
            second = result.pop()
            first = result.pop()
            result.append(and_query(first, second, in_memory_dictionary, posting_list_file))
        
        return result[0]
        
def print_result(result):
    res = ''
    curr = result.head
    while(curr is not None):
        res += str(curr.data) + ' '
        curr = curr.next
    return res.strip()

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    with open(dictionary_file, 'rb') as f:
        in_memory_dictionary = pickle.load(f)
        N = pickle.load(f)
        final_calculated_normalised_length = pickle.load(f)

    posting_list_file = open(postings_file, 'rb')

    count = 0

    with open(queries_file, 'r') as file:
        queries = file.readlines()
        relevant_docs = []
        for query in queries:
            print("FOR")
            print(query)
            if(count == 0):
                print('s')
                result = process_query(query, N, in_memory_dictionary, posting_list_file, final_calculated_normalised_length)
            else:
                print('d')
                relevant_docs.append(queries) # KIV
            # count += 1
    
    results = open(results_file, "w")
    results.write(print_result(result))
    results.close()
    
"""
    results = open(results_file, "w")

    for query in queries:
        tokens = get_term(query)
        cosine_without_normalisation = {} # docId: second vector, where second vector is a dict {mappingid: 1 + logtf term query value}
        curr_query_term_freq = {} # maps mappingid to [term freq in entire query, df in index] (for first vector - query)
        curr_query_term_freq_mapping = {} # maps term to number, where number = mappingid

        current_counter = 0
        
        for token in tokens:
            if(token in in_memory_dictionary):
                seek_val = in_memory_dictionary[token][1]
                posting_list_file.seek(seek_val)
                posting_list = pickle.load(posting_list_file)

                if (token in curr_query_term_freq_mapping):
                    mappingid = curr_query_term_freq_mapping[token]
                    curr_query_term_freq[mappingid][0] += 1
                elif (token not in curr_query_term_freq_mapping):
                    query_term_df = posting_list.df
                    curr_query_term_freq_mapping[token] = current_counter
                    mappingid = current_counter
                    current_counter += 1
                    curr_query_term_freq[mappingid] = [1, query_term_df]
                
                    # calculation of first (query) vector in dot product(for 'cosine_without_normalisation')
                    curr = posting_list.head

                    while(curr is not None):
                        currDocId = curr.data
                        if (currDocId not in cosine_without_normalisation):
                            cosine_without_normalisation[currDocId] = {}
                        
                        cosine_without_normalisation[currDocId][mappingid] = 1 + math.log10(curr.tf)
                        curr = curr.next

        # if no matching documents, return early and continue to next query
        if (len(cosine_without_normalisation) == 0):
            results.write('\n')
            continue

        # processing curr_query_term_freq to calculate tf-idf for each unique term in query
        for key, value in curr_query_term_freq.items():
            tf = 1 + math.log10(value[0])
            idf = math.log10(N/value[1])
            tf_idf = tf * idf
            # update value of each key to tf_idf value
            curr_query_term_freq[key] = tf_idf

        # start calculating cosine similarity for each document
        normalised_query_length = 0
        for key, value in curr_query_term_freq.items():
            normalised_query_length += value**2
        normalised_query_length = math.sqrt(normalised_query_length)

        for key, value in cosine_without_normalisation.items():
            counter = 0
            for key2, value2 in value.items(): # doing dot product before normalising
                counter += curr_query_term_freq[key2] * value2 # query val * doc val for common terms between query and doc
            
            # divide by the normalising length in 'final_calculated_normalised_length' (for doc vec length)
            counter /= final_calculated_normalised_length[key] # key = docId
            
            # divide by the normalising length, normalised_query_length (for query vec length)
            counter /= normalised_query_length
            
            # take normalised counter value and replace it with value in 'cosine_without_normalisation'
            cosine_without_normalisation[key] = counter
        
        # take top k (10 most relevant (less if there are fewer than ten documents that have matching stems to the query) docIDs in response to the query))
        heap = [(-value, key) for key, value in cosine_without_normalisation.items()]
        largest = heapq.nsmallest(10, heap)
            
        query_write_output = [key for value, key in largest]

        # write to results (output file)
        results.write(str(query_write_output[0]))

        for docId in query_write_output[1:]:
            results.write(' ')
            results.write(str(docId))

        if (query != queries[-1]):
            results.write('\n')
        
    results.close()
"""

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
