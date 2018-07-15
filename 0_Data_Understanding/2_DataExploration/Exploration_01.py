# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:58:07 2017

@author: Wei Jing
"""

# Data Exploration 01-A
# Get a sense of what are the features in Malaysia Accident Case Data

# Initialise all library imports
import pandas
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Initilise all global variables
gExcelCorpus = {} # Create new dictionary
gExcelColumn1 = [] # Create new array
gExcelColumn2 = [] # Create new array
gExcelColumn3 = [] # Create new array

# Function to load and Pre-Process Data
def fLoadPP(vExcelFile):
    # Initialise all local variables
    vColumn1Len = 0
    vColumn2Len = 0
    vColumn3Len = 0
    vColumn1Str = ""
    vColumn2Str = ""
    vColumn3Str = ""
    
    # Load Excel Corpus
    gExcelCorpus = pandas.read_excel(vExcelFile)
    
    # Tokenise Columns
    #vC1Tokens = word_tokenize(str(gExcelCorpus['Cause']))
    #vC2Tokens = word_tokenize(str(gExcelCorpus['Title Case']))
    #vC3Tokens = word_tokenize(str(gExcelCorpus['Summary Case']))
    
    
    # Store in Array 1
    vColumn1Len = len(gExcelCorpus['Cause'])
    for i in range (vColumn1Len):
        gExcelColumn1.append(gExcelCorpus['Cause'][i])
    
    # Store in Array 2
    vColumn2Len = len(gExcelCorpus['Title Case'])
    for j in range (vColumn2Len):
        gExcelColumn2.append(gExcelCorpus['Title Case'][j])
        
    # Store in Array 3
    vColumn3Len = len(gExcelCorpus['Summary Case'])
    for k in range (vColumn3Len):
        gExcelColumn3.append(gExcelCorpus['Summary Case'][k])
    
    # Convert the Free Text into Tokens
    vC1Tokens = word_tokenize(str(gExcelColumn1))
    vC2Tokens = word_tokenize(str(gExcelColumn2))
    vC3Tokens = word_tokenize(str(gExcelColumn3))
    
    # Unique Words
    #vC3Unique = set(vC3Tokens)
    
    # Remove Punctuations
    vTokens_nop = [ t for t in vC3Tokens if t not in string.punctuation]
    
    # Convert All Lower Case
    vTokens_lower=[ t.lower() for t in vTokens_nop]
    
    # Create Stopwords
    vStopWords = stopwords.words('english')
	
	#select english stopwords
    vStopWords = set(stopwords.words("english"))
    #add custom word
    vStopWords.update(('victim', '\'the'))
    #remove stop words
    #new_str = ' '.join([word for word in str.split() if word not in cachedStopWords]) 
    
    # Remove All Stopwords From Text
    vTokens_nostop=[ t for t in vTokens_lower if t not in vStopWords]
    
    # Apply Porter Stemmer
    vPorter = nltk.PorterStemmer()
    vTokens_porter=[vPorter.stem(t) for t in vTokens_nostop] 
    
    # Apply Lancaster Stemmer 
    vLancaster = nltk.LancasterStemmer()
    vTokens_lanc = [vLancaster.stem(t) for t in vTokens_nostop] 
    
    # Apply Snowball Stemmer
    vSnowball = nltk.SnowballStemmer('english')
    vTokens_snow = [vSnowball.stem(t) for t in vTokens_nostop]
    
    # Remove words with less then or equals 3 characters
    vTokens_clean = [t for t in vTokens_snow if len(t) >= 4]
    
    # Frequency Distribution of the Words
    vFrequencyDistri = nltk.FreqDist(vTokens_clean)
    vFrequencyDistri.most_common(30)
    vFrequencyDistri.plot(30)
    
    # Join the cleaned tokens back into a string
    vText_clean=" ".join(vTokens_clean)
    
    # 1. Simple cloud 
    # Generate a word cloud image
    # Take note that this function requires text string as input
    vWordCloud_Simple_01 = WordCloud(background_color="white").generate(vText_clean)
    
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(vWordCloud_Simple_01, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    vWordCloud_Simple_01.to_file("example_Malaysia_01.png")
    
    #print(gExcelCorpus['Cause'])
    print(vText_clean)
    
print("Total Jobs")
fLoadPP('MsiaAccidentCases.xlsx')

# Function to load and Pre-Process Data (OSHA)
def fOSHALoadPP(vExcelFile):
    # Initialise all local variables
    vColumn1Len = 0
    vColumn2Len = 0
    vColumn3Len = 0
    vColumn1Str = ""
    vColumn2Str = ""
    vColumn3Str = ""
    
    # Load Excel Corpus
    gExcelCorpus = pandas.read_excel(vExcelFile)
    
    # Tokenise Columns
    #vC1Tokens = word_tokenize(str(gExcelCorpus['Cause']))
    #vC2Tokens = word_tokenize(str(gExcelCorpus['Title Case']))
    #vC3Tokens = word_tokenize(str(gExcelCorpus['Summary Case']))
    
    
    # Store in Array 1
    vColumn1Len = len(gExcelCorpus['NO'])
    for i in range (vColumn1Len):
        gExcelColumn1.append(gExcelCorpus['NO'][i])
    
    # Store in Array 2
    vColumn2Len = len(gExcelCorpus['Title Case'])
    for j in range (vColumn2Len):
        gExcelColumn2.append(gExcelCorpus['Title Case'][j])
        
    # Store in Array 3
    vColumn3Len = len(gExcelCorpus['Summary Case'])
    for k in range (vColumn3Len):
        gExcelColumn3.append(gExcelCorpus['Summary Case'][k])
    
    # Convert the Free Text into Tokens
    vC1Tokens = word_tokenize(str(gExcelColumn1))
    vC2Tokens = word_tokenize(str(gExcelColumn2))
    vC3Tokens = word_tokenize(str(gExcelColumn3))
    
    # Unique Words
    #vC3Unique = set(vC3Tokens)
    
    # Remove Punctuations
    vTokens_nop = [ t for t in vC3Tokens if t not in string.punctuation]
    
    # Convert All Lower Case
    vTokens_lower=[ t.lower() for t in vTokens_nop]
    
    # Create Stopwords
    vStopWords = stopwords.words('english')
    
    # Remove All Stopwords From Text
    vTokens_nostop=[ t for t in vTokens_lower if t not in vStopWords]
    
    # Apply Porter Stemmer
    vPorter = nltk.PorterStemmer()
    vTokens_porter=[vPorter.stem(t) for t in vTokens_nostop] 
    
    # Apply Lancaster Stemmer 
    vLancaster = nltk.LancasterStemmer()
    vTokens_lanc = [vLancaster.stem(t) for t in vTokens_nostop] 
    
    # Apply Snowball Stemmer
    vSnowball = nltk.SnowballStemmer('english')
    vTokens_snow = [vSnowball.stem(t) for t in vTokens_nostop]
    
    # Remove words with less then or equals 3 characters
    vTokens_clean = [t for t in vTokens_snow if len(t) >= 4]
    
    # Frequency Distribution of the Words
    vFrequencyDistri = nltk.FreqDist(vTokens_clean)
    vFrequencyDistri.most_common(30)
    vFrequencyDistri.plot(30)
    
    # Join the cleaned tokens back into a string
    vText_clean=" ".join(vTokens_clean)
    
    # 1. Simple cloud 
    # Generate a word cloud image
    # Take note that this function requires text string as input
    vWordCloud_Simple_01 = WordCloud(background_color="white").generate(vText_clean)
    
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(vWordCloud_Simple_01, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    vWordCloud_Simple_01.to_file("example_OSHA_01.png")
    
    #print(gExcelCorpus['Cause'])
    print(vText_clean)

fOSHALoadPP('osha.xlsx')
