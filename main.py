import  spacy
from spacy.lang.en.stop_words import STOP_WORDS
import csv
import pandas as pd
from nltk.corpus import stopwords
import re
import os
import numpy as np
#-------------------------------------------------Input location----------------------------------------------------------------------#

INPUT_CSV = "/Users/sethuram/Desktop/test.csv"

conversation_reader = csv.reader(open(INPUT_CSV), delimiter=',')

# -------------------------------------------------Data Cleaning----------------------------------------------------------------------#


read_csv  = list();
filtered_data = []
stop_word_removed_data = []


for row in conversation_reader:
    read_csv.append(row[0].split("\t"))

print(read_csv);
for i in read_csv:
    if(i[2]=="Participant"):
        temp = re.sub("<.*>"," ",i[3]).strip()
        if(temp != ""):
            filtered_data.append(temp);

print("The Size of Filterd Data is ",len(filtered_data))
#print(filtered_data)

#--------STOP WORD REMOVAL-----------------------#

nlp = spacy.load("en")
STOP_WORDS = set(("""mmm hmm um uh""").split())
# STOP_WORDS.add("mmm,hmm".split(","))
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True


#test=['I am  fine and life is cool','I will enjoy', 'I am happy']

# iterating text
for data in filtered_data:
    doc = nlp(data)
    text = []
    for w in doc:
        if not w.is_stop:
            text.append(w.lemma_)
    text_str = " ".join(text)
    stop_word_removed_data.append(text_str);

print("The Size of Stop Word Removed Data is ", len(stop_word_removed_data));
#print(stop_word_removed_data)

#--------Google Cloud Connection-----------------------#

def could_hitter():
    from google.cloud import language

    #-------client connection-------#
    GCLOUD_CREDS_PATH = "/Users/sethuram/Desktop/Google/My First Project-ea40b9924211.json"

    def get_client():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCLOUD_CREDS_PATH
        return language.LanguageServiceClient();

    return get_client();


global google_client;
google_client = could_hitter();


def sentiment_analyzer(text):
    from google.cloud.language import enums
    from google.cloud.language import types
    from google.api_core.exceptions import InvalidArgument
    import six

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(content=text,type=enums.Document.Type.PLAIN_TEXT)
    sentiment = google_client.analyze_sentiment(document).document_sentiment
    #print("score",sentiment.score,"magnitude",sentiment.magnitude)
    return(sentiment.score *sentiment.magnitude ,sentiment.score,sentiment.magnitude);




def score_percentage_caluculator(mean_score_for_negative_emotion):
    return(((mean_score_for_negative_emotion + 0.25)/(-0.75))*100);

def magnitude_percentage_caluculator(mean_magnitude_for_negative_emotion):
    if(mean_magnitude_for_negative_emotion >1):
        return 100
    return(mean_magnitude_for_negative_emotion*100);

def overall_depression_lines_percentage_calculator(count):
    return((count/len(stop_word_removed_data))*100)



def depression_calculators(count,mean_score_for_negative_emotion,mean_magnitude_for_negative_emotion):
    return((overall_depression_lines_percentage_calculator(count)+(score_percentage_caluculator(mean_score_for_negative_emotion)/2))+(magnitude_percentage_caluculator(mean_magnitude_for_negative_emotion)/4));

def sentiment_analysis_initiator(conversations):
    count = 0;
    overall_score=list();
    overall_magnitude = list();
    for each_conversation in conversations:
         overall_emotion,score,magnitude  = sentiment_analyzer(each_conversation)
         if(overall_emotion <0):
            count = count+1
            overall_score.append(score)
            overall_magnitude.append(magnitude)
         if(len(overall_magnitude) == 0):
             return "He is Happy"

    mean_score_for_negative_emotion = np.mean(overall_score);
    mean_magnitude_for_negative_emotion = np.mean(overall_magnitude);
    return depression_calculators(count,mean_score_for_negative_emotion,mean_magnitude_for_negative_emotion);



def level_of_depression(conversations):
    overall_depression_percentage = sentiment_analysis_initiator(conversations);
    if(isinstance(overall_depression_percentage, str)):
        print(overall_depression_percentage)
    else:
        print(overall_depression_percentage);
        if (0 <= overall_depression_percentage <= 25):
            print("neutral");
        elif (25 <= overall_depression_percentage <= 50):
            print("low depressed");
        elif (50 <= overall_depression_percentage <= 75):
            print("moderate depressed");
        elif (75 <= overall_depression_percentage <= 100):
            print("Highly depressed");


level_of_depression(stop_word_removed_data);
#print(stop_word_removed_data)


