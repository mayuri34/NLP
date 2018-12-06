A pythom program to train probabilistic language models and distinguish between words in different languages. This specific implementation calculates accuracy of 'Universal Declaration of Human Rights' in English vs French and Spanish vs Italian.

Program: 
1. Divide dataset of each language into dev, train and test sets. 
2. Pre-process training dataset by removing punctuations and new line characters(\n) 
3. Create character level unigrams, bigrams and trigrams of each training dataset 
4. Calculate their frequencies using nltk.FreqDist() 
5. Calculate probabilities of common n-grams in test set and training set of two languages by,    
	∑ P(frequency distance of current n-gram)  
6. Keep a count of higher probability between both the languages 
7. Accuracy = first language’s probability score / total characters in test set 


Output:

Accuracy for unigram in English and French is: 0.785 

Accuracy for bigram in English and French is:  0.91 

Accuracy for trigram in English and French is: 0.993 

Accuracy for unigram in Spanish and Italian is: 0.671 

Accuracy for bigram in Spanish and Italian is:  0.838 

Accuracy for trigram in Spanish and Italian is: 0.98 

 
 