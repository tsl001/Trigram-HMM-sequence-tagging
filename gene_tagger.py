#! /usr/bin/python

import sys
from collections import defaultdict
import math
import random
import gensim
from gensim.models import Word2Vec
import heapq
import itertools
import string

def counts_corpus_iterator(corpus_file, with_logprob = False):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()    
    tagfield = with_logprob and -2 or -1
    
    try:
        
        while l:
            line = l.strip()
            #print(line)
            if line: # Nonempty line
                # Extract information from line.
                # Each line has the format
                # word ne_tag [log_prob]
                fields = line.split(" ")

                if fields[1] == "WORDTAG":
                    ne_tag = fields[-2]
                    word = fields[-1] #" ".join(fields[:tagfield])
                    count = fields[0]
                    yield "WORDTAG",ne_tag,word,count,None
                elif fields[1] == "1-GRAM":
                    count = fields[0]
                    tag1 = fields[-1]
                    yield "1-GRAM",tag1,count,None,None
                elif fields[1] == "2-GRAM":
                    count = fields[0]
                    tag1 = fields[-2]
                    tag2 = fields[-1]
                    yield "2-GRAM",tag1,tag2, count, None
                elif fields[1] == "3-GRAM":
                    count = fields[0]
                    tag1 = fields[-3]
                    tag2 = fields[-2]
                    tag3 = fields[-1]
                    yield "3-GRAM",tag1,tag2,tag3,count

            # else: # Empty line
            #     yield (None, None)
            l = corpus_file.readline()
    except IndexError:
        sys.stderr.write("Could not read line: \n")
        sys.stderr.write("\n%s" % line)
        if with_logprob:
            sys.stderr.write("Did you forget to output log probabilities in the prediction file?\n")
        sys.exit(1)

def corpus_iterator(corpus_file, with_logprob = False):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    #print("hi")
    l = corpus_file.readline()    
    
    try:
        
        while l:
            line = l.strip()
            #print(line)
            if line: # Nonempty line
                # Extract information from line.
                # Each line has the format
                # word ne_tag [log_prob]
                #fields = line.split(" ")
                yield line
            else: # Empty line
                yield "\n"
            l = corpus_file.readline()
    except IndexError:
        sys.stderr.write("Could not read line: \n")
        sys.stderr.write("\n%s" % line)
        if with_logprob:
            sys.stderr.write("Did you forget to output log probabilities in the prediction file?\n")
        sys.exit(1)

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag

            # fields = line.split(" ")
            # ne_tag = fields[-1]

            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            #word = " ".join(fields[:-1])
            yield line #word, ne_tag
        else: # Empty line
            yield None                       
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def usage():
    sys.stderr.write("""
    Usage: python eval_gene_tagger.py [key_file] [prediction_file]
        Evaluate the gene-tagger output in prediction_file against
        the gold standard in key_file. Output accuracy, precision,
        recall and F1-Score.\n""")

class Hmm(object):
    def __init__(self,emission_counts=defaultdict(int), tag_counts=defaultdict(int),one_grams_counts=defaultdict(int),
                 two_grams_counts=defaultdict(int),three_grams_counts=defaultdict(int),q_params_dict=defaultdict(float)):
        self.emission_counts = emission_counts
        self.tag_counts = tag_counts
        self.possible_tags = ['O','I-GENE']
        self.one_grams_counts = one_grams_counts
        self.two_grams_counts = two_grams_counts
        self.three_grams_counts = three_grams_counts #(tag_{i-2},tag_{i-1},tag_{i})
        self.q_params_dict = q_params_dict
        self.word2vecModel = None
        #self.initializeWord2VecModel()
        

    def initializeWord2VecModel(self,devIteratorList):
        wordsList = []
        for word in devIteratorList:
            wordsList.append(word)
        wordsList = [wordsList]

        self.word2vecModel = gensim.models.Word2Vec(wordsList,min_count = 1, window = 5)

    def getMaxTag(self,word):
        #for k in self.emission_counts.keys() if k[0] == word
        
        tagProbs = defaultdict(float)
        
        for tag in self.possible_tags:
            if (word,tag) in self.emission_counts:
                tagProbs[tag] = self.emission_counts[(word,tag)] / self.tag_counts[tag]
        
        if len(tagProbs.keys()) > 0:
            predTag = max(tagProbs,key=tagProbs.get)
            return predTag #will return the maximum tag
        elif len(tagProbs.keys()) == 0: #not seen so RARE word tag
            if word.isdigit():
                return 'O'

            if word[:3] == 'tri':
                return 'O'

            if word[:4] == 'endo':
                return 'O'

            if word[:3] == 'bio':
                return 'O'
            
                
            rareTagProb = defaultdict(float)
            if ('_RARE_','O') in self.emission_counts:
                rareTagProb['O'] = self.emission_counts[('_RARE_','O')] / self.tag_counts['O']
            
            if ('_RARE_','I-GENE') in self.emission_counts:
                rareTagProb['I-GENE'] = self.emission_counts[('_RARE_','I-GENE')] / self.tag_counts['I-GENE']
            
            predTag = max(rareTagProb,key=rareTagProb.get)
            return predTag

    def getEmissionProb(self,word,tag):
        if (word,tag) in self.emission_counts:
            return self.emission_counts[(word,tag)] / self.tag_counts[tag]
        elif tag == 'I-GENE' and (word,'O') in self.emission_counts:
            return 0 #self.emission_counts[('_RARE_',tag)] / self.tag_counts[tag]
        elif tag == 'O' and (word,'I-GENE') in self.emission_counts:
            return 0
        elif (word,'I-GENE') not in self.emission_counts and (word,'O') not in self.emission_counts:
            # heapq.nlargest(5, self.emission_counts, key=self.emission_counts.get)
            
            # onlyDigitsEmissionCounts = dict()
            # for k in self.emission_counts.keys():
            #     if k[0].isdigit():
            #         onlyDigitsEmissionCounts[k] = self.emission_counts[k] / self.tag_counts[k[1]]

            # mostLikelyOnlyDigitsKeys = heapq.nlargest(100, onlyDigitsEmissionCounts, key=onlyDigitsEmissionCounts.get)
            # print(mostLikelyOnlyDigitsKeys[:20],file=sys.stderr)

            if word.isdigit():
                if tag == 'O':
                    return self.emission_counts[('1','O')]/self.tag_counts['O']
                else:
                    return self.emission_counts[('1','I-GENE')]/self.tag_counts['I-GENE']



            # if word[:3] == 'tri':
            #     return self.emission_counts[('_RARE_','O')] / self.tag_counts['O']

            # if word[:4] == 'endo':
            #     return self.emission_counts[('_RARE_','O')] / self.tag_counts['O']

            # if word[:3] == 'bio':
            #     return self.emission_counts[('_RARE_','O')] / self.tag_counts['O']

            rareTagProb = defaultdict(float)
            if ('_RARE_','O') in self.emission_counts:
                rareTagProb['O'] = self.emission_counts[('_RARE_','O')] / self.tag_counts['O']
            
            if ('_RARE_','I-GENE') in self.emission_counts:
                rareTagProb['I-GENE'] = self.emission_counts[('_RARE_','I-GENE')] / self.tag_counts['I-GENE']
            
            predTag = max(rareTagProb,key=rareTagProb.get)
            return rareTagProb[predTag] #self.emission_counts[('_RARE_',tag)] / self.tag_counts[tag]#


    def calculate_param_q(self,trigram):
        bigram = (trigram[0],trigram[1])
        #print(trigram)
        return self.three_grams_counts[trigram] / self.two_grams_counts[bigram]

    def getBestOmega(self,trigram, pi_kuv_combinations,idx): #omega = w
        tag_i = trigram[2]
        tag_i1 = trigram[1]
        #tag_i2 = trigram[0]

        #emissionProb = getEmissionProb(word,tag_i)
        q_probs = defaultdict(float)
        

        for tagToTry in self.possible_tags:
            q_probs[(tagToTry,tag_i1,tag_i)] = pi_kuv_combinations[(idx - 1, tagToTry, tag_i1)] + math.log(q_params_dict[(tagToTry,tag_i1,tag_i)],2)
                
        bestTrigramKey = max(q_probs,key=q_probs.get)

        #return emissionProb * q_probs[bestTrigramKey]
        return bestTrigramKey[0]

        

def baseline(model):
    dev_iterator = corpus_iterator(open(sys.argv[2]))
    #print(len(list(dev_iterator)),file=sys.stderr

    dev_iterator = corpus_iterator(open(sys.argv[2]))
    

    for word in dev_iterator:
        if word == "\n":
            print()
        else:
            predTag = model.getMaxTag(word)
            print(word + " " + predTag)

def writeViterbiSeq(seqToWrite):
    dev_iterator = corpus_iterator(open(sys.argv[2]))
    idx = 0
    
    for word in dev_iterator:
        if word == "\n":
            print()
        else:
            print(word + " " + seqToWrite[idx])
            idx += 1


def viterbi(model,q_params_dict):
    dev_iterator = corpus_iterator(open(sys.argv[2]))
    #dev_iterator_list = list(dev_iterator)

    bp_kuv_combinations = defaultdict(str)
    pi_kuv_combinations = defaultdict(float)
    idx = 0

    #for idx,word in enumerate(dev_iterator_list):
    for word in dev_iterator: #while (word := next(dev_iterator, None)) is not None:
        if word != "\n":
            for v in model.possible_tags:
                for u in model.possible_tags:
                    if idx == 0:
                        emissionProb = model.getEmissionProb(word,v)
                        if emissionProb == 0:
                            max_pi_calculation = math.log(1,2) + math.log(q_params_dict[('*','*',v)],2) + float('-inf')
                        else:
                            max_pi_calculation = math.log(1,2) + math.log(q_params_dict[('*','*',v)],2) + math.log(emissionProb,2)

                        pi_kuv_combinations[(idx,'*',v)] = max_pi_calculation
                        bp_kuv_combinations[(idx,'*',v)] = '*' #bestOmega
                    elif idx == 1:
                        emissionProb = model.getEmissionProb(word,v)
                        if emissionProb == 0:
                            max_pi_calculation = pi_kuv_combinations[(idx - 1,'*',u)] + math.log(q_params_dict[('*',u,v)],2) + float('-inf')
                        else:
                            max_pi_calculation = pi_kuv_combinations[(idx - 1,'*',u)] + math.log(q_params_dict[('*',u,v)],2) + math.log(emissionProb,2)

                        pi_kuv_combinations[(idx,u,v)] = max_pi_calculation
                        bp_kuv_combinations[(idx,u,v)] = '*' #bestOmega
                    else:
                        bestOmega = model.getBestOmega((None,u,v), pi_kuv_combinations, idx) #[(idx - 1,None,u)])
                        emissionProb = model.getEmissionProb(word,v)
                        if emissionProb == 0:
                            max_pi_calculation = pi_kuv_combinations[(idx - 1,bestOmega,u)] + math.log(q_params_dict[(bestOmega,u,v)],2) + float('-inf')
                        else:
                            max_pi_calculation = pi_kuv_combinations[(idx - 1,bestOmega,u)] + math.log(q_params_dict[(bestOmega,u,v)],2) + math.log(emissionProb,2)

                        pi_kuv_combinations[(idx,u,v)] = max_pi_calculation
                        bp_kuv_combinations[(idx,u,v)] = bestOmega
                        
            idx += 1 #increment index

    lastIdx = idx - 1
    
    retArgs = []
    bestStopPi = defaultdict(float)

    for v in model.possible_tags:
        for u in model.possible_tags:
            bestStopPi[(u,v,'STOP')] = pi_kuv_combinations[(lastIdx,u,v)] + math.log(q_params_dict[(u,v,'STOP')],2)

    
    bestStopPiKey = max(bestStopPi,key=bestStopPi.get)
    retArgs.append(bestStopPiKey[1])
    retArgs.append(bestStopPiKey[0]) #append v first then u
    #print(retArgs)

   # y_plus_2 = bestStopPiKey[1]
    #y_plus_1 = bestStopPiKey[0]

    for idx in range(lastIdx-2,-1,-1):
        y_plus_1 = retArgs[-1]
        y_plus_2 = retArgs[-2]

        #print(idx+2, file=sys.stderr)
        appendArg = bp_kuv_combinations[(idx + 2,y_plus_1,y_plus_2)]
        retArgs.append(appendArg)

    retArgs.reverse()
    return retArgs
    


if __name__ == "__main__":
    if len(sys.argv)!=3:
        usage()
        sys.exit(1)
    
    
    counts_iterator = counts_corpus_iterator(open(sys.argv[1]))
    tag_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    one_grams_counts = defaultdict(int)
    two_grams_counts = defaultdict(int)
    three_grams_counts = defaultdict(int)
    q_params_dict = defaultdict(float)

    for this_type,param1,param2,param3,param4 in counts_iterator:
        if this_type == "WORDTAG":
            if param1 not in tag_counts:
                tag_counts[param1] = int(param3)
            else:
                tag_counts[param1] += int(param3)

            if (param2,param1) not in emission_counts: #param1 is tag and param2 is word
                emission_counts[(param2,param1)] = int(param3) #param3 is counts
            else:
                emission_counts[(param2,param1)] += int(param3)
        elif this_type == "1-GRAM":
            one_grams_counts[param1] = int(param2) #param2 is count param1 is tag
        elif this_type == "2-GRAM":
            two_grams_counts[(param1,param2)] = int(param3) 
        elif this_type == "3-GRAM":
            three_grams_counts[(param1,param2,param3)] = int(param4)
            if (param1,param2,param3) not in q_params_dict:
                q_params_dict[(param1,param2,param3)] = int(param4) / two_grams_counts[(param1,param2)]
        
    model = Hmm(emission_counts=emission_counts,tag_counts=tag_counts,one_grams_counts=one_grams_counts,two_grams_counts=two_grams_counts,
                three_grams_counts=three_grams_counts,q_params_dict=q_params_dict)

    #baseline(model)
    retSeq = viterbi(model,q_params_dict)
    writeViterbiSeq(retSeq)
    
        

        

        