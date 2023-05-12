#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import os
import pickle

from nltk.tokenize import word_tokenize
from nltk import stem
from nltk.corpus import stopwords
import math
import time
import heapq

from postlist import *

stemmer = stem.PorterStemmer()

courts_to_check = ["SG Court of Appeal", "SG Privy Council", "UK House of Lords", "UK Supreme Court",
                    "High Court of Australia", "CA Supreme Court", "SG High Court", "Singapore International Commercial Court", "HK High Court",
                "HK Court of First Instance", "UK Crown Court", "UK Court of Appeal", "UK High Court", "Federal Court of Australia",
                "NSW Court of Appeal", "NSW Court of Criminal Appeal", "NSW Supreme Court"]

courts_to_check_lower = []
# python3 search.py -d dictionary_file10 -p postings_file10 -q q1.txt -o output-file-of-results

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

def decode_varbyte_string(encoded):
    values = []
    i = 0
    while i < len(encoded):
        value = 0
        shift = 0
        while True:
            byte = encoded[i]
            value |= (byte & 0x7F) << shift
            i += 1
            if byte & 0x80 == 0:
                break
            shift += 7
        values.append(value)
    return values

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

def preprocess_text(text_to_process):
    text_to_process = text_to_process.replace('\n', ' ')
    text_to_process = ''.join(char for char in text_to_process if char.isalnum() or char.isspace())
    return text_to_process

def phrase_helper_getRelevantDocs(first, second, in_memory_dictionary, posting_list_file): #returns a pl of relevant docs
    print("here")
    print(in_memory_dictionary[first])
    temp_df1, temp_pickled_index1, unique_id = in_memory_dictionary[first]
    posting_list_file.seek(temp_pickled_index1)
    temp_pl1 = pickle.load(posting_list_file)
    temp_df2, temp_pickled_index2, unique_id = in_memory_dictionary[second]
    posting_list_file.seek(temp_pickled_index2)
    temp_pl2 = pickle.load(posting_list_file)
    
    common_docIds = and_query_on_tuple_phrase_query(temp_pl1, temp_pl2, in_memory_dictionary, posting_list_file)
    
    return common_docIds
    
#if overall query is a boolean query
def phrase_query(phrase_query, in_memory_dictionary, posting_list_file):
    tokens = get_term(phrase_query)
    print(tokens)
    print(len(tokens))
    relevant_docs = PostingList()
    #preliminary check that all tokens exist in in_memory_dictionary, else return empty posting list
    for token in tokens:
        if token not in in_memory_dictionary: #word: [df, pickle index]
            return relevant_docs

    temp_postlist = -1 #intialise with placeholder value -1
    for i in range(len(tokens) - 1): #on the condition that all tokens exist in the in_memory_dictionary
        token1 = tokens[i]
        token2 = tokens[i + 1]
        
        if (temp_postlist == -1):
            temp_postlist = phrase_helper_getRelevantDocs(token1, token2, in_memory_dictionary, posting_list_file)
            temp_postlist.addSkipPointer()
        else: #temp_postlist is already a posting list with documentids
            curr_postlist = phrase_helper_getRelevantDocs(token1, token2, in_memory_dictionary, posting_list_file)
            curr_postlist.addSkipPointer()
            temp_postlist = and_query(temp_postlist, curr_postlist, in_memory_dictionary, posting_list_file)
            temp_postlist.addSkipPointer()
    
    return temp_postlist

def find_results(query, in_memory_dictionary, posting_list_file):
    tokens = get_term(query)
    print("check: tokens")
    print(tokens)
    cosine_without_normalisation = {} # docId: second vector, where second vector is a dict {mappingid: 1 + logtf term query value}
    curr_query_term_freq = {} # maps mappingid to [term freq in entire query, df in index] (for first vector - query)
    curr_query_term_freq_mapping = {} # maps term to number, where number = mappingid

    current_counter = 0
# 6807771
# 4001247
# 3992148
          
    for token in tokens:
        if(token in in_memory_dictionary):
            seek_val = in_memory_dictionary[token][1]
            posting_list_file.seek(seek_val)
            posting_list = pickle.load(posting_list_file)

            if (token in curr_query_term_freq_mapping):
                mappingid = curr_query_term_freq_mapping[token]
                curr_query_term_freq[mappingid][0] += 1
            elif (token not in curr_query_term_freq_mapping):
                # query_term_df = posting_list.df
                query_term_df = len(posting_list)
                curr_query_term_freq_mapping[token] = current_counter
                mappingid = current_counter
                current_counter += 1
                curr_query_term_freq[mappingid] = [1, query_term_df]
            
                # calculation of first (query) vector in dot product(for 'cosine_without_normalisation')
        
                for curr_pointer in range(len(posting_list)):
                    curr = posting_list[curr_pointer]
                    currDocId = curr[0]
                    if (currDocId not in cosine_without_normalisation):
                        cosine_without_normalisation[currDocId] = {}
                    
                    cosine_without_normalisation[currDocId][mappingid] = 1 + math.log10(curr[1])
    
    return curr_query_term_freq, cosine_without_normalisation, curr_query_term_freq_mapping

def calculate_tf_idf(N, curr_query_term_freq, cosine_without_normalisation, final_calculated_normalised_length):
    # processing curr_query_term_freq to calculate tf-idf for each unique term in query
    # print("curr_query_term_freq")
    # print(curr_query_term_freq)
    for key, value in curr_query_term_freq.items():
        tf = 1 + math.log10(value[0])
        idf = math.log10(N/value[1])
        #CHECK: IDF < 0.1
        if (idf < 0.1):
            idf = 0
        tf_idf = tf * idf
        # update value of each key to tf_idf value
        curr_query_term_freq[key] = tf_idf
    # print(curr_query_term_freq)
    # start calculating cosine similarity for each document
    
    normalised_query_length = 0
    for key, value in curr_query_term_freq.items():
        normalised_query_length += value**2
    normalised_query_length = math.sqrt(normalised_query_length)
    
    for key, value in cosine_without_normalisation.items():
        counter = 0
        for key2, value2 in value.items(): # doing dot product before normalising
            counter += curr_query_term_freq[key2] * value2 # query val * doc val for common terms between query and doc
        
        '''
        # divide by the normalising length in 'final_calculated_normalised_length' (for doc vec length)
        counter /= final_calculated_normalised_length[key] # key = docId
        
        # divide by the normalising length, normalised_query_length (for query vec length)
        counter /= normalised_query_length
        '''
        
        # take normalised counter value and replace it with value in 'cosine_without_normalisation'
        cosine_without_normalisation[key] = counter
    
    # sorted_result = sorted(cosine_without_normalisation.items(), key=lambda x:x[1], reverse = True)

    # result = []
    # values = []
    # for item in sorted_result:
    #     result.append(item[0])
    #     values.append(item[1])
    
    # print(values)

    # return result
    # print("score_dict")
    # print(cosine_without_normalisation)
    return cosine_without_normalisation

def and_query_on_tuple_phrase_query(first, second, in_memory_dictionary, p):
    if (type(first) != tuple):
        seek_value_first = in_memory_dictionary[first][1]
        p.seek(seek_value_first)
        first = pickle.load(p)
    if (type(second) != tuple):
        seek_value_second = in_memory_dictionary[second][1]
        p.seek(seek_value_second)
        second = pickle.load(p)
    to_traverse = first
    other = second
    skip_to_traverse = math.floor(math.sqrt(len(to_traverse)))
    skip_other = math.floor(math.sqrt(len(other)))
    pointer_to_traverse = 0
    pointer_other = 0
    docId_to_traverse = to_traverse[pointer_to_traverse]
    docId_other = other[pointer_other]
    common_docIds = PostingList()
    while (pointer_to_traverse < len(to_traverse) and pointer_other < len(other)):
        if (docId_to_traverse < docId_other):
            #advance to_traverse
            if (pointer_to_traverse % skip_to_traverse == 0 and pointer_to_traverse + skip_to_traverse < len(to_traverse)
                and to_traverse[pointer_to_traverse + skip_to_traverse] <= docId_other):
                pointer_to_traverse += skip_to_traverse
                docId_to_traverse = to_traverse[pointer_to_traverse]
            else:
                pointer_to_traverse += 1
                if (pointer_to_traverse < len(to_traverse)):
                    docId_to_traverse = to_traverse[pointer_to_traverse]
        elif (docId_to_traverse == docId_other):
            #check if positional index fulfils phrase criteria of being apart by one
            to_traverse_positional_index = to_traverse[pointer_to_traverse][2]
            other_positional_index = other[pointer_other][2]
            if (type(to_traverse_positional_index) != tuple or type(other_positional_index) != tuple):
                raise KeyError
            else:
                for i in to_traverse_positional_index:
                    decoded_value = decode_varbyte(i)
                    if encode_varbyte(decoded_value + 1) in other_positional_index:
                        common_docIds.addNode(docId_to_traverse)
                        break
                pointer_other += 1
                pointer_to_traverse += 1
                if (pointer_to_traverse < len(to_traverse)):
                    docId_to_traverse = to_traverse[pointer_to_traverse]
                if (pointer_other < len(other)):
                    docId_other = other[pointer_other]
                
            
        elif (docId_to_traverse > docId_other):
            if (pointer_other % skip_other == 0 and pointer_other + skip_other < len(other)
                and other[pointer_other + skip_other] <= docId_to_traverse):
                pointer_other += skip_other
                docId_other = other[pointer_other]
            else:
                pointer_other += 1
                if (pointer_other < len(other)):
                    docId_other = other[pointer_other]
    return common_docIds
        

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

def helperSearch(query, in_memory_dictionary, posting_list_file, final_calculated_normalised_length, court_mapping, court_score_mapping, zones_and_fields_dict, N):
    set_rel_docs_overall = -1
    #preliminary filter for phrasal queries
    query_after_processing = ""

    if "AND" in query:
        queries = query.split(" AND ")
        for each_query in queries:
            #preprocess queries
            processed_query = ''.join(char for char in each_query if char.isalnum() or char.isspace())
            query_after_processing += processed_query
            query_after_processing += ' '
            if each_query[0] == each_query[-1] and each_query[0] == '"' or each_query[0] == "'":
                if set_rel_docs_overall == -1:
                    set_rel_docs_overall = phrase_query(processed_query, in_memory_dictionary, posting_list_file)
                else:
                    set_rel_docs_for_curr_phrase = phrase_query(processed_query, in_memory_dictionary, posting_list_file)
                    set_rel_docs_overall = and_query(set_rel_docs_for_curr_phrase, set_rel_docs_overall, in_memory_dictionary, posting_list_file)
            else:
                #process the rest as free text(?) (e.g. hello in "i am a phrase" AND hello)
                continue
                        
        query_after_processing = query_after_processing[:-1] #to delete last spacing

            
    else:
        query_after_processing = ''.join(char for char in query if char.isalnum() or char.isspace())
        
    curr_query_term_freq, cosine_without_normalisation, curr_query_term_freq_mapping = find_results(query_after_processing, in_memory_dictionary, posting_list_file)
    # print("curr_query_term_freq")
    # print(curr_query_term_freq)
    score_dict = calculate_tf_idf(N, curr_query_term_freq, cosine_without_normalisation, final_calculated_normalised_length) #at this point in time, score of each relevant docId is just tf-idf
        
    docs_not_selected_in_prelim_filter = []
    docs_selected_in_prelim_filter = []
        
    if (set_rel_docs_overall != -1): #means boolean query initially (preliminary filter done)
        #make set_rel_docs_overall as an array instead of posting list
        set_rel_docs_overall_asArray = []
        curr = set_rel_docs_overall.head
        while curr is not None:
            set_rel_docs_overall_asArray.append(curr.data)
            curr = curr.next
                
        for each_result in set_rel_docs_overall_asArray:
            if each_result in score_dict:
                score_dict[each_result] += 0.01 #arbitrary bonus score for being in prelim filter
            
    modifyScoreByTitle_Court_Date(score_dict, query, court_mapping, court_score_mapping, zones_and_fields_dict)
    return score_dict, curr_query_term_freq, curr_query_term_freq_mapping
        

def run_search(dictionary_file, postings_file, query_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    startTime = time.time()
    
    ###change all court names to lowercase
    for court in courts_to_check:
        courts_to_check_lower.append(court.lower())
    ###
           
    with open(dictionary_file, 'rb') as f:
        in_memory_dictionary = pickle.load(f)
        full_docIds = pickle.load(f)
        N = len(full_docIds)
        # print("total num docs", N)
        final_calculated_normalised_length = pickle.load(f)
        
        zones_and_fields_dict = pickle.load(f)
        court_mapping = pickle.load(f)
        court_score_mapping = pickle.load(f)
        # court_name_to_docId_mapping = pickle.load(f)
        relevance_query_dict = pickle.load(f)
        
        # print("RELEVANCE QUERY DICT")
        # print(relevance_query_dict)
    # print(len(in_memory_dictionary))
    posting_list_file = open(postings_file, 'rb')

    count = 0
    
    with open(query_file, 'r') as file:
        temp = file.read().splitlines()
        query = temp[0]
        relevant_docs = temp[1:] #given
        # print(relevant_docs)
        score_dict, curr_query_term_freq, curr_query_term_freq_mapping = helperSearch(query, in_memory_dictionary, posting_list_file, final_calculated_normalised_length, court_mapping, court_score_mapping, zones_and_fields_dict, N)
        results = sort_score_dict_by_score_descending(score_dict)
    
        ###perform relevance query if len(relevant_docs) > 0
        if (len(relevant_docs) > 0):
            reformulated_query = perform_relevance_query(curr_query_term_freq, curr_query_term_freq_mapping, relevant_docs, in_memory_dictionary, relevance_query_dict, N)
            if (reformulated_query != False):
                score_dict, curr_query_term_freq, curr_query_term_freq_mapping = helperSearch(reformulated_query, in_memory_dictionary, posting_list_file, final_calculated_normalised_length, court_mapping, court_score_mapping, zones_and_fields_dict, N)
                results = sort_score_dict_by_score_descending(score_dict)
        
        """
        ###find non_relevant_docs_from_initial_search and relevant_docs_from_initial_search for rocchio's later
        non_relevant_docs_from_intial_search = []
        length_non_relevant_docs_from_initial_search = 0
        for docid in full_docIds:
            if length_non_relevant_docs_from_initial_search < 2000:
                if docid not in results:
                    non_relevant_docs_from_intial_search.append(docid)
            else:
                break
            
        if length_non_relevant_docs_from_initial_search < 2000:
            #take from end of results
            diff = 2000 - length_non_relevant_docs_from_initial_search
            non_relevant_docs_from_initial_search += results[-diff:]
        """                                    
        # if len(results) > 1700:
        #     results = results[:1700]
        
        """
        ###use rocchio's formula here
        relevant_docs_from_initial_search = results[:2000]
        use_rocchio(relevant_docs)
        ###
        """
    
    f_results = open(results_file, "w")
    for answer in results:
        f_results.write(str(answer))
        f_results.write(" ")
    f_results.close()
    
    print ("Execution Time:" + str(time.time() - startTime) + "s")

###relevance query (not so sure)
def perform_relevance_query(curr_query_term_freq, curr_query_term_freq_mapping, relevant_docs, in_memory_dictionary, relevance_query_dict, N):
    #tunable weights, weight_alpha + weight_beta == 1
    weight_alpha = 0.25
    weight_beta = 1 - weight_alpha
    num_rel_docs = len(relevant_docs)
    
    num_unique_terms_index = len(in_memory_dictionary)
    term_to_index_mapping = {} #{"hi": 3} means "hi" is the index 3 of query_vector and rel_docs_vector
    query_vector = [0] * (num_unique_terms_index)
    
    next_avail_id = 0
    
    normalised_length_query = 0
    temp_storage_query = {}
    for term, id in curr_query_term_freq_mapping.items():
        term_to_index_mapping[term] = id
        tf_idf_value = curr_query_term_freq[id] #gets unnormalised tf-idf value
        normalised_length_query += tf_idf_value**2
        temp_storage_query[id] = tf_idf_value
        next_avail_id += 1 #might be tricky
    
    normalised_length_query = math.sqrt(normalised_length_query)
        
    for id, value in temp_storage_query.items():
        normalised_tfIdf = value
        if (normalised_length_query > 0):
            normalised_tfIdf = value/normalised_length_query
        query_vector[id] = normalised_tfIdf

    rel_docs_vector = [0] * (num_unique_terms_index)
    #added sorted in_memory_dict    
    in_memory_dictionary_sorted = dict(sorted(in_memory_dictionary.items(), key=lambda item: item[1][2]))
    in_memory_dictionary_sorted_list = [(key, value) for key, value in in_memory_dictionary_sorted.items()]
    for curr_rel_doc in relevant_docs:
    
        curr_rel_doc_int = int(curr_rel_doc) #convert from string to integer
        #ACCOUNT FOR WHEN CURR_REL_DOC_INT NOT IN RELEVANCE_QUERY_DICT
        if curr_rel_doc_int in relevance_query_dict:
            continue
        tuple_of_uniqueWordId_termFreqs = relevance_query_dict[curr_rel_doc_int]
        temp_storage_relDocs = {} #temp storage of tf-idf values before normalisation. so we can add to rel_docs_vector after normalising
        normalisation_counter = 0
        for var_byte_string in tuple_of_uniqueWordId_termFreqs: #unique words in each doc
            ls = decode_varbyte_string(var_byte_string)
            if (len(ls) == 1):
                id = 0
                tf = ls[0]
            elif (len(ls) == 2):
                id, tf = ls
            else:
                raise ValueError
            word = in_memory_dictionary_sorted_list[id][0]
            df = in_memory_dictionary[word][0]
            curr_tfIdf = relevance_query_calculate_tf_idf_nonNormalised(tf, df, N)
            temp_storage_relDocs[word] = curr_tfIdf
            normalisation_counter += (curr_tfIdf)**2
        
        final_normalised_length_for_curr_doc = math.sqrt(normalisation_counter)
        
        #loop through temp_storage since now we can add normalised tf-idf values
        for word, value in temp_storage_relDocs.items():
            normalised_tf_idf = value
            if (final_normalised_length_for_curr_doc > 0):
                normalised_tf_idf = value/final_normalised_length_for_curr_doc
            if word in term_to_index_mapping:
                get_id = term_to_index_mapping[word]
                rel_docs_vector[get_id] += normalised_tf_idf
            else:
                term_to_index_mapping[word] = next_avail_id
                rel_docs_vector[next_avail_id] += normalised_tf_idf
                next_avail_id += 1
            
    max_index_in_vectors = next_avail_id - 1
    query_vector_processed = query_vector[:max_index_in_vectors + 1]
    rel_docs_vector_processed = rel_docs_vector[:max_index_in_vectors + 1]
    resultant_vector = [0] * (max_index_in_vectors + 1)
    
    for i in range(max_index_in_vectors + 1): # i is curr_index
        resultant_vector[i] = (query_vector_processed[i] * weight_alpha) + ((rel_docs_vector_processed[i]/num_rel_docs) * weight_beta)
    
    final_reconstructed_query = ""
    # reconstructed_query_dict = {}
    value_list = []
    index_to_term_mapping = sorted(term_to_index_mapping.items(), key=lambda x: x[1])
    for i in range(len(resultant_vector)):
        term = index_to_term_mapping[i][0]
        value = resultant_vector[i]
        # reconstructed_query_dict[term] = value
        value_list.append((value, term))
    
    # heuristic: take top (max 20 unique) values
    heapq.heapify(value_list)
    how_many = min(20, len(value_list))
    top_values = heapq.nlargest(how_many, value_list)
    print(top_values)
    min_value = top_values[how_many - 1][0]
    while min_value == 0:
        how_many -= 1
        top_values = heapq.nlargest(how_many, value_list)
        min_value = top_values[how_many - 1][0]
    for curr_value, term in top_values:
        if (min_value > 0):
            num_times = int(round(curr_value/min_value))
        #jic:
        if (num_times == 0):
            num_times = 1
            
        for i in range(num_times):
            final_reconstructed_query += term
            final_reconstructed_query += ' '
    
    #jic
    if (final_reconstructed_query == ""):
        return False
    
    print(final_reconstructed_query)
    return final_reconstructed_query
    
    
    print(curr_query_term_freq, curr_query_term_freq_mapping, relevant_docs)
    print("done")

def relevance_query_calculate_tf_idf_nonNormalised (tf, df, N):
        tf = 1 + math.log10(tf)
        idf = math.log10(N/df)
        #CHECK: IDF < 0.1
        if (idf < 0.1):
            idf = 0
        tf_idf = tf * idf
        return tf_idf
    
##### note: zones_and_fields_dict mapping: zones_and_fields_dict[docID] = (title_tuple, court.lower(), extracted_date)
def modifyScoreByTitle_Court_Date(score_dict, query, court_mapping, court_score_mapping, zones_and_fields_dict):
    bonus_date_score = 0.015
    for docId, currScore in score_dict.items():
        # add bonus court score
        query_processed_for_court_checking = query.lower()
        required_court_name = zones_and_fields_dict[docId][1]
        if required_court_name in query_processed_for_court_checking:
            court_id = court_mapping[required_court_name]
            court_importance_score = court_score_mapping[court_id]
            if court_importance_score == 1:
                bonus_court_score = 0.015
            else:
                bonus_court_score = 0.02
            score_dict[docId] += bonus_court_score
        
        # add bonus title score        
        query_processed_for_title_checking = preprocess_text(query)
        tokens_query = get_term(query_processed_for_title_checking)
        unique_tokens_query = list(set(tokens_query))
        ###find title score
        title_tuple = zones_and_fields_dict[docId][0]
        len_title_tuple = len(title_tuple)
        num_common_terms = 0
        
        for token in unique_tokens_query:
            if token in title_tuple:
                num_common_terms += 1
        percentage_common_with_title = num_common_terms/len_title_tuple
        bonus_title_score = percentage_common_with_title * 0.15
        score_dict[docId] += bonus_title_score
        
        ###find date score
        query_processed_for_date_checking = query.lower()
        required_date = zones_and_fields_dict[docId][2]
        if required_date in query_processed_for_date_checking:
            score_dict[docId] += bonus_date_score
            

def sort_score_dict_by_score_descending(score_dict):
    sorted_result = sorted(score_dict.items(), key=lambda x:x[1], reverse = True)

    resulting_docIds = []
    score = []
    for item in sorted_result:
        resulting_docIds.append(item[0])
        score.append(item[1])
    print(len(score))
    # print(score[0], score[5000], score[-1])

    return resulting_docIds

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
