import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import random
import string, re, unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia as wk

#The text is extracted from Wikipedia https://en.wikipedia.org/wiki/Artificial_intelligence
f = open('ai.txt','r',errors = 'ignore')
txt = f.read()
txt = txt.lower()

sent_tokens = nltk.sent_tokenize(txt)
print(sent_tokens[:2])

#Responses for greetings inputs
GREETING_INPUTS = ("hello", "hi", "hey", "hey there","greetings", "what's up",)
GREETING_RESPONSES = ["Hi", "Hey", "Hey there", "Hello","Hola"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    word_tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    
    #Remove ascii
    new_words = []
    for word in word_tokens:
        new_word = unicodedata.normalize(u'NFKD', word).encode('ascii', 'ignore').decode('utf-8')
        new_words.append(new_word)
    
    #Remove tags
    rmv = []
    for w in new_words:
        text=re.sub("&lt;/?.*?&gt;","&lt;&gt;",w)
        rmv.append(text)
        
    #POS tagging and lemmatization
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list

def generateResponse(inputMessage):
    botResponse=''
    sent_tokens.append(inputMessage)
    TfidfVec = TfidfVectorizer(tokenizer=normalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0) or "tell me about" in inputMessage:
        print("Checking Wikipedia..")
        if inputMessage:
            botResponse = wikipediaSearch(inputMessage)
            return botResponse
    else:
        botResponse = botResponse+sent_tokens[idx]
        return botResponse

def wikipediaSearch(input):
    regex = re.search('tell me about (.*)', input)
    try:
        if regex:
            topic = regex.group(1)
            wiki = wk.summary(topic, sentences = 2)
            return wiki
    except Exception as e:
            print(e)

print("Chatbot: Hey there! I can answer your questions about Artificial Intelligence. Type \"Bye\" to exit")
flag=True
while(flag==True):
    inputMessage = input()
    inputMessage = inputMessage.lower()
    if(inputMessage!='bye'):
        if(inputMessage=='thanks' or inputMessage=='thank you' ):
            flag=False
            print("Chatbot: You are welcome..")
        elif(greeting(inputMessage)!=None):
                print("Chatbot: " + greeting(inputMessage))
        else:
            print("Chatbot: ",end="")
            print(generateResponse(inputMessage))
            sent_tokens.remove(inputMessage)
    else:
        flag=False
        print("Chatbot:bye!")