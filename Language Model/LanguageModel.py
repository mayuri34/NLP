import nltk
import re
import string
from nltk.util import ngrams
from nltk.corpus import udhr  
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

english = udhr.raw('English-Latin1')
french = udhr.raw('French_Francais-Latin1')
italian = udhr.raw('Italian_Italiano-Latin1')
spanish = udhr.raw('Spanish_Espanol-Latin1') 

english_train, english_dev = english[0:1000], english[1000:1100]
french_train, french_dev = french[0:1000], french[1000:1100]
italian_train, italian_dev = italian[0:1000], italian[1000:1100]
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]  

english_test = udhr.words('English-Latin1')[0:1000]
french_test = udhr.words('French_Francais-Latin1')[0:1000]
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000]
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

def pre_process(text):   
    clean_text = ''.join([(re.sub('\\n',' ', word)) for word in text.lower() if word not in string.punctuation])
    return clean_text
	
def create_freq_dist(text, n):    
    fdist = nltk.FreqDist(w for w in list(ngrams(text, n)))       
    return fdist
	
def calculate_accuracy(first_language, second_language, test, n):
    fl_score = 0
    freqdist_fl_ngram_model = create_freq_dist(first_language,n)
    freqdist_sl_ngram_model = create_freq_dist(second_language,n)
    lower_test = [ch.lower() for ch in test]
    for ch in lower_test:
        fl_probability = 1
        sl_probability = 1
        for text in [ngrams(ch, n)]:
            for t in text:                
                fl_probability = fl_probability * freqdist_fl_ngram_model[t]
                sl_probability = sl_probability * freqdist_sl_ngram_model[t]
                    
            if fl_probability >= sl_probability:
                fl_score = fl_score+1
    return fl_score / len(lower_test)
	
clean_english_train = pre_process(english_train)
clean_french_train = pre_process(french_train)
clean_spanish_train = pre_process(spanish_train)
clean_italian_train = pre_process(italian_train)

report = []
report.append('------Question 1------')
accuracy = calculate_accuracy(clean_english_train, clean_french_train, english_test, 1)
report.append('Accuracy for unigram in English and French is : ')
report.append(accuracy)

accuracy = calculate_accuracy(clean_english_train, clean_french_train, english_test, 2)
report.append('Accuracy for bigram in English and French is : ')
report.append(accuracy)

accuracy = calculate_accuracy(clean_english_train, clean_french_train, english_test, 3)
report.append('Accuracy for trigram in English and French is : ')
report.append(accuracy)

report.append('------Question 2------')
accuracy = calculate_accuracy(clean_spanish_train, clean_italian_train, spanish_test, 1)
report.append('Accuracy for unigram in Spanish and Italian is : ')
report.append(accuracy)

accuracy = calculate_accuracy(clean_spanish_train, clean_italian_train, spanish_test, 2)
report.append('Accuracy for bigram in Spanish and Italian is : ')
report.append(accuracy)

accuracy = calculate_accuracy(clean_spanish_train, clean_italian_train, spanish_test, 3)
report.append('Accuracy for trigram in Spanish and Italian is : ')
report.append(accuracy)

for line in report:
    print(line)
