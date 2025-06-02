# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Project Part 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
#
# This code will be graded automatically using Gradescope.
# =========================================================================================================
import nltk
from nltk.corpus import treebank
import numpy as np
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Function: get_treebank_data
# Input: None
# Returns: Tuple (train_sents, test_sents)
#
# This function fetches tagged sentences from the NLTK Treebank corpus, calculates an index for an 80-20 train-test split,
# then splits the data into training and testing sets accordingly.

def get_treebank_data():
    # Fetches tagged sentences from the NLTK Treebank corpus.
    sentences = nltk.corpus.treebank.tagged_sents()
    # Calculate the split index for an 80-20 train-test split.
    split = int(len(sentences) * 0.8)
    # Split the data into training and testing sets.
    train_sents = sentences[:split]
    test_sents = sentences[split:]
    return train_sents, test_sents

# Function: compute_tag_trans_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary A of tag transition probabilities
#
# Iterates over training data to compute the probability of tag bigrams (transitions from one tag to another).

# The transition probability matrix A represents the prob of a tag occuring given previous tag
# MLE: p(t_i|t_i-1) = count(t_i-1, t_i)/count(t_i-1)
def compute_tag_trans_probs(train_data):
    tag_counts = {}
    transition_counts = {}
    # Extract all unique tags
    unique_tags = set()

    for sent in train_data:
        tags = [pair[1] for pair in sent]  # Extract tags from word-tag pairs
        unique_tags.update(tags)

        for i in range(len(tags) - 1):
            # count occurence of tag i and the following tag
            transition_counts[(tags[i], tags[i + 1])] = transition_counts.get((tags[i], tags[i + 1]), 0) + 1
            tag_counts[tags[i]] = tag_counts.get(tags[i], 0) + 1

        tag_counts[tags[-1]] = tag_counts.get(tags[-1], 0) + 1  # Count last tag

    # Initialize transition matrix with zero probabilities
    A = {tag1: {tag2: 0 for tag2 in unique_tags} for tag1 in unique_tags}

    # Convert counts to probabilities
    for (tag1, tag2), count in transition_counts.items():
        A[tag1][tag2] = count / tag_counts[tag1]

    return A  # Returns a dictionary-based matrix

# Function: compute_emission_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary B of tag-to-word emission probabilities
#
# Iterates through each sentence in the training data to count occurrences of each tag emitting a specific word, then calculates probabilities.

# Emission probability matrix B: probability of tag associated with a gien word p(w_i|ti)
# MLE: p(w_i|t_i) = count(t_i, w_i)/count(t_i)
def compute_emission_probs(train_data):
    emission_counts = {}
    tag_counts = {}

    # Iterate through each sentence to count word-tag pairs and individual tags
    for sent in train_data:
        for word, tag in sent:
            emission_counts[(tag, word)] = emission_counts.get((tag, word), 0) + 1
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Convert counts to probabilities
    B = {}
    for (tag, word), count in emission_counts.items():
        if tag not in B:
            B[tag] = {}
        B[tag][word] = count / tag_counts[tag]

    return B

# Function: viterbi_algorithm
# Input: words (list of words that have to be tagged), A (transition probabilities), B (emission probabilities)
# Returns: List (the most likely sequence of tags for the input words)
#
# Implements the Viterbi algorithm to determine the most likely tag path for a given sequence of words, using given transition and emission probabilities.

def viterbi_algorithm(words, A, B):
    states = list(B.keys())
    Vit = [{}]
    path = {}

    ## Initialization for t=0
    for state in states:
        Vit[0][state] = B.get(state, {}).get(words[0], 0.0001)  # Handle unknown words - assign small prob
        path[state] = [state]

    # Implement Viterbi for t > 0
    for t in range(1, len(words)):
        # initialize empty vectors
        Vit.append({})
        new_path = {}
        for curr_state in states:
            (prob, prev_state) = max(
                (Vit[t-1][prev_state] * A.get((prev_state, curr_state), 0.0001) * B.get(curr_state, {}).get(words[t], 0.0001), prev_state)
                for prev_state in states
            )

            Vit[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Find the most probable final state
    (final_prob, final_state) = max((Vit[len(words) - 1][state], state) for state in states)

    return path[final_state]

# Function: evaluate_pos_tagger
# Input: test_data (tagged sentences for testing), A (transition probabilities), B (emission probabilities)
# Returns: Float (accuracy of the POS tagger on the test data)
#
# Evaluates the POS tagger's accuracy on a test set by comparing predicted tags to actual tags and calculating the percentage of correct predictions.

def evaluate_pos_tagger(test_data, A, B):
    correct = 0
    total = 0
    # Iterate through each sentence in the test data
    for sent in test_data:
        words = [word for word, tag in sent]  # Extract words from sentence
        actual_tags = [tag for word, tag in sent]  # Extract actual tags
        # Predict tags
        predicted_tags = viterbi_algorithm(words, A, B)
        # Compare predicted tags with actual tags
        correct += sum(1 for pred, actual in zip(predicted_tags, actual_tags) if pred == actual)
        total += len(actual_tags)  # Total number of words
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    # Main function to train and evaluate the POS tagger.
    

    train_data, test_data = get_treebank_data()
    A = compute_tag_trans_probs(train_data)
    B = compute_emission_probs(train_data)

    # Print specific probabilities
    print(f"P(VB -> DT): {A['VB'].get('DT', 0):.4f}") # Expected Probability should be checked 0.2296
    print(f"P(DT -> 'the'): {B['DT'].get('the', 0):.4f}")  # Expected Probability should be checked 0.4986
    
    # Evaluate the model's accuracy
    accuracy = evaluate_pos_tagger(test_data, A, B)
    print(f"Accuracy of the HMM-based POS Tagger: {accuracy:.4f}") ## Expected accuracy around 0.8743



