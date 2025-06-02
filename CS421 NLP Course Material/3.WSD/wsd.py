# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Assignment 3
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import nltk
from nltk.corpus import wordnet as wn

#Function to load and return sentences from the semcor library
#num: the number of sentences to load
#Returns: first num sentences from semcor
from nltk.corpus import semcor
def load_sentences(num):
    #sents = None
    #YOUR CODE HERE
    sents = semcor.sents()[0:num]
    return sents

#Function to load and return tagged sentences from the semcor library
#num: the number of sentences to load
#Returns: first num tagged sentences from semcor
def load_tagged_sents(num):
    #labels = None
    #YOUR CODE HERE
    labels = semcor.tagged_sents(tag="sem")[0:num]
    return labels


#DO NOT MODIFY THIS FUNCTION
#Function to process the labels and get the wordnet synset for words
#sentences: the tagged sentences to be processed
#Returns: first num tagged sentences from semcor
def process_labels(sentences):
    sents = []
    labels = []
    for sent in sentences:  # Iterate through the first 5 sentences for demonstration
        curr_sent = []
        curr_labels = []
        sense_words = []
        for word in sent:
            if isinstance(word, nltk.Tree):  # Check if it is a tree (has sense)
                lemma = word.label()  # Get the sense label
                text = "_".join(word.leaves())  # Get the word(s) corresponding to this sense
                try:
                    if 'group.n.' not in lemma.synset().name(): # Do not add if it is a group of proper nouns
                        curr_sent.append(text)
                        curr_labels.append(lemma.synset().name())
                except:
                    curr_sent.append(text)
                    curr_labels.append(lemma)
        sents.append(curr_sent)
        labels.append(curr_labels)
    return sents, labels
# This function process the input sentences, and extract the true lables of each words
# the true label comes from the semcor.tagged_sents() wrapped in nltk.Tree()
# For a give sentence, each word only has 1 true label as it has been annotated

#Function to get the word sense for a given word using the most frequent word sense in wordnet
#word: the word for which the sense is to be calculated
#Returns: the name of the synset for the calculated sense
def most_freq_sense_model(word):
    #sense = None
    #YOUR CODE HERE
    syn = wn.synsets(word)
    if syn == []:
        return None
    sense = syn[0].name()
    return sense

#Function to run the most_freq_sense model on all sentences
#sentences: List of list of strings containing words which make up the sentences to get predictions on
#Return: list of list of predicted senses
def get_most_freq_predictions(sentences):
    #preds = None
    #YOUR CODE HERE
    preds = []
    for sent in sentences: # for 1 sentence in all sentences
        sent_pred = []
        for words in sent: # for 1 word in the 1 sentence
            sense = most_freq_sense_model(words)
            sent_pred.append(sense)
        preds.append(sent_pred)
    return preds


import nltk
from nltk.corpus import wordnet as wn, stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
# define stopwords to filter out
stop_words = set(stopwords.words("english"))
#Function to calculate the length of overlap between 2 sentences
#after tokenization and after removing the stopwords
#return: the number of words overlapping between two sentences
def ComputeOverlap(signature, context):
    # if statements to avoid empty string
    signature = set(signature.split()) - stop_words if signature else set()
    context = set(context.split()) - stop_words if context else set()
    return len(signature & context)  # Count common words

#Function to get the word sense for a given word using the most frequent word sense in wordnet
#word: the word for which the sense is to be calculated
#sentence: the sentence in which the word is used
#Returns: the name of the synset for the calculated sense
def lesk_model(word, sentence):
    if not isinstance(sentence, str):
        sentence = " ".join(sentence)  # Convert list of words to string if needed

    best_sense = wn.synsets(word)[0].name() if wn.synsets(word) else None  # Default to most frequent sense
    max_overlap = 0
    context = set(sentence.lower().split()) - stop_words  # Convert sentence to a set of words. Remove stopwords

    for sense in wn.synsets(word):
        signature = set()  # Initialize signature as an empty set
        # Get definition words if exist
        definition = sense.definition() if sense.definition() else ""
        # add definition to the signature
        signature.update(definition.lower().split())
        # Get example words if exist
        for example in sense.examples():
            if example:  # Ensure example is not None
                # add the example words to the signature
                signature.update(example.lower().split())
        # calculate number of overlaps
        overlap = ComputeOverlap(" ".join(signature), " ".join(context))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense.name()  # Update best sense
            # to extract the definition of the sense
            #best_sense_definition = sense.definition()
    return best_sense


#Function to run the lesk on all sentences
#sentences: List of list of strings containing words which make up the sentences to get predictions on
#Return: list of list of predicted senses
def get_lesk_predictions(sentences):
    #preds = None
    #YOUR CODE HERE
    preds = []
    for sent in sentences: # for 1 sentence in all sentences
        sent_pred = []
        full_sentence = " ".join(sent) # put words back to sentence
        for words in sent: # for 1 word in the 1 sentence
            sense = lesk_model(words, full_sentence)
            sent_pred.append(sense if sense else None)
        preds.append(sent_pred)
    return preds

#Function to evaluate the predictions
#labels: List of list of strings containing the actual senses
#predicted: List of list of strings containing the predicted senses
#Return: precision, recall and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score
def evaluate(labels, predicted):
    p = None
    r = None
    f1 = None
    #YOUR CODE HERE
    true_label = []
    pred_label = []
    for actual, predicted in zip(labels, predicted):
        for act, prd in zip(actual, predicted):
            if act != None and prd != None:
                true_label.append(act)
                pred_label.append(prd)
    p = precision_score(true_label, pred_label, average='weighted', zero_division=0)
    r = recall_score(true_label, pred_label, average='weighted', zero_division=0)
    f1 = f1_score(true_label, pred_label, average='weighted', zero_division=0)
    return p, r, f1

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python wsd.py

def main():
    # Download WordNet and SemCor data if not already downloaded
    nltk.download('wordnet')
    nltk.download('semcor')
    #Load the sentences and tagged sentences
    sents = load_sentences(50)
    tagged_sents = load_tagged_sents(50)
    #Process the tagged sentences to get the labels
    processed_sentences, labels = process_labels(tagged_sents)
    #Get the predictions using most frequent sense model
    preds_mfs = get_most_freq_predictions(processed_sentences)
    #Evaluate the predictions on the most frequent sense model
    print(evaluate(labels, preds_mfs))
    #Get the predictions using lesk model
    preds_lesk = get_lesk_predictions(processed_sentences)
    #Evaluate the predictions on the lesk model
    print(evaluate(labels, preds_lesk))
    
if __name__ == '__main__':
    main()