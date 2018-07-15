# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:42:12 2017

@author: Wei Jing
"""
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import svm
import pickle
import csv
import math
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import string

gUpdateStopwords = ["victim", "suddenly", "believed", "hospital", "time", "place", "scene", "approximately", "victims", "confirmed", "happened"]
gPunctuations = [".", ",", '"', "'", "?", "!", ":", ";", "(", ")", "[", "]", "{", "}", "@", "#", "$", "%", "^", "&", "*", "-", "_", "+", "=", "/", "\\", "|", "<", ">"]
gNumberList = ["1", "0", "2", "3", "4", "5", "6", "7", "8", "9"]
gSemanticCommonActivities = {}

# -----------------------------------------------------------------------------
# Load Dictionary
# -----------------------------------------------------------------------------

def gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary):
    
    #--------------------------------------------------------------------------
    # Load Dictionary
    vDictionaryDataFrame = pd.read_csv(vLoadCSV_Dictionary, encoding="ISO-8859-1")
    
    #--------------------------------------------------------------------------
    # Get Column data to load in Dictionary
    vTempDictionary = {}
    vFinalDictionary = {}
    vFeaturesLength = len(vLoad_Features)
    
    for i in range(vFeaturesLength):
        vTempDictionary[vLoad_Features[i]] = vDictionaryDataFrame[str(vLoad_Features[i])]
    
    #--------------------------------------------------------------------------
    # Test if obj is instance of string
    # Return dictionary object without nan
    for j in vTempDictionary:
        vTempLength = len(vTempDictionary[j])
        vTempArray = []
        
        for k in range(vTempLength):
            if isinstance(vTempDictionary[j][k], str):
                #print(vTempDictionary[j][k])
                vTempArray.append(vTempDictionary[j][k])
        
        vFinalDictionary[j] = vTempArray
    
    #print("Compiled Dictionary :", vFinalDictionary)
    
    return vFinalDictionary

# -----------------------------------------------------------------------------
# Association Rules
# Function to populate categories based on associated terms
# -----------------------------------------------------------------------------

def gAssociation2Category(vCategory, vAssociationTermDictionary, vMultipleAssociation, vInput, vOutput):
    
    vCaseDictionary = {}
    vFeaturesDictionary = {}
    vTempScores = []
    
    #--------------------------------------------------------------------------
    # Get comparable Case data
    # Load input in CaseDictionary
    # vInput assume to only have two columns Case_Title and Cases
    
    # Load Case_Title and Cases
    vCasesDataFrame = pd.read_csv(vInput)
    vCaseDictionary["Case_Title"] = vCasesDataFrame["Case_Title"]
    vCaseDictionary["Cases"] = vCasesDataFrame["Cases"]
    
    #--------------------------------------------------------------------------
    # Score each features by comparing occurence in each case
    # Treat each feature as a word in a case
    # where vTempScore[Feature] = [score for each case]
    
    # Step 1
    # Set default scores
    
    vCaseDictionaryLength = len(list(vCaseDictionary["Case_Title"]))
    
    for i in range(vCaseDictionaryLength):
        vTempScores.append(0)
    
    # Step 2
    # Create Features Library
    # Go through each cases, check if feature exists
    for j in vAssociationTermDictionary:
        vFeatureTokens = list(vAssociationTermDictionary[j])
        vTokensLength = len(vFeatureTokens)
        
        for k in range(vTokensLength):
            vFeaturesDictionary[str(vFeatureTokens[k])] = vTempScores
    
    for l in vFeaturesDictionary:
        vTempArray = []
        vTempArray = list(vFeaturesDictionary[l])
        vTempLength = len(vTempArray)

        for m in range(vTempLength):
            vCaseTitleString = vCaseDictionary["Case_Title"][m]
            vCaseString = vCaseDictionary["Cases"][m]
            
            #vWord = " " + str(l) + " "
            #------------------------------------------------------------------
            vWord = str(l)
            
            vCheckWord = str(vCaseTitleString)
            vPunctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''#This code defines punctuation
            # This code removes the punctuation
            
            vNoPunctuations = "" 
            for vChar in vCheckWord:
                if vChar not in vPunctuations:
                    vNoPunctuations = vNoPunctuations + vChar
            
            vNoPunctuations = vNoPunctuations.lower()
            
            vStart_Index = str(vNoPunctuations).find(vWord)
            vEnd_Index = vStart_Index + len(vWord) # if the start_index is not -1
            vStringLength = len(str(vNoPunctuations))

            # Check if vWord exists as the 1st Word in String
            if vStart_Index == 0:
                vWord = vWord + " "
            elif vEnd_Index == vStringLength:
                vWord = " " + vWord
            elif vStart_Index > 1:
                vWord = " " + vWord + " "
            
            if str(vWord) in str(vNoPunctuations):
                vArray = []
                vArray = list(vFeaturesDictionary[l])
                vArray[m] = 1
                vFeaturesDictionary[l] = vArray
                #print(vWord, "    :", vNoPunctuations)
            
            #------------------------------------------------------------------
            vWord = str(l)
            
            vCheckWord = str(vCaseString)
            vPunctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''#This code defines punctuation
            # This code removes the punctuation
            
            vNoPunctuations = "" 
            for vChar in vCheckWord:
                if vChar not in vPunctuations:
                    vNoPunctuations = vNoPunctuations + vChar
            
            vNoPunctuations = vNoPunctuations.lower()
            
            vStart_Index = str(vNoPunctuations).find(vWord)
            vEnd_Index = vStart_Index + len(vWord) # if the start_index is not -1
            vStringLength = len(str(vNoPunctuations))

            # Check if vWord exists as the 1st Word in String
            if vStart_Index == 0:
                vWord = vWord + " "
            elif vEnd_Index == vStringLength:
                vWord = " " + vWord
            elif vStart_Index > 1:
                vWord = " " + vWord + " "
            
            if str(vWord) in str(vNoPunctuations):
                vArray = []
                vArray = list(vFeaturesDictionary[l])
                vArray[m] = 1
                vFeaturesDictionary[l] = vArray
       
    #--------------------------------------------------------------------------
    # Defining the category type based on the features and their score for each case    
    # Rule 1 - if more than one category has a score - Extra category (Multiple)
    # Rule 2 - if no category has a score > 0 - Extra category (Nonclassificable)
    # Rule 3 - if one category has a score - Category
    # where vFinalCategory = [category for each case]
    
    vFinalCategory = []
    
    for i in range(vCaseDictionaryLength):
        vFinalCategory.append("NonClassifiable")
    
    vCatCount = {}
    
    for i in range(vCaseDictionaryLength):
        vCounter = 0
        vLabel = "NonClassifiable"
        vMultipleLabels = []
        for j in vFeaturesDictionary:
            if vFeaturesDictionary[j][i] == 1:
                for k in vAssociationTermDictionary:
                    vCatCount[str(k)] = 0
                    vTempArray = list(vAssociationTermDictionary[k])
                    vTempLength = len(vTempArray) 
                    for l in range(vTempLength):
                        if str(vTempArray[l]) == str(j):
                            if vCounter == 0:
                                vLabel1 = str(k)
                                vCounter += 1
                                vMultipleLabels.append(k)
                                vFinalCategory[i] = vLabel1
                            elif vCounter > 0:
                                #vLabel = "Multiple_Parts"
                                vCounter += 1
                                vMultipleLabels.append(k)
                        
        vMultipleLength = len(vMultipleLabels)

        if vMultipleLength > 1:
            for z in range(vMultipleLength):
                for y in vAssociationTermDictionary:
                    if str(vMultipleLabels[z]) == str(y):
                        vCatCount[str(y)] += 1
        
        vBigger = ""
        
        for x in vCatCount:
            if vBigger == "":
                vBigger = x
            elif vBigger != "":
                if vCatCount[x] > vCatCount[vBigger]:
                    vBigger = x
        
        vLabel = vBigger
        
        if vFinalCategory[i] != "NonClassifiable":
            if vMultipleLength > 1:
                vFinalCategory[i] = vLabel
        elif vFinalCategory[i] == "NonClassifiable":
            if vMultipleAssociation == "BP":
                vBodySystemArray = ["contusion", "contusions", "burn", "burns", "abrasion", "abrasions"]
                vLengthBodySystemArray = len(vBodySystemArray)
                
                for w in range(vLengthBodySystemArray):
                    # If nonclassifiable check for the word skin
                    vCaseTitleString = vCaseDictionary["Case_Title"][i]
                    vCaseString = vCaseDictionary["Cases"][i]
                    
                    #--------------------------------------------------------------
                    vWord = str(vBodySystemArray[w])
                    vCheckWord = str(vCaseTitleString)
                    vPunctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''#This code defines punctuation
                    # This code removes the punctuation
                
                    vNoPunctuations = "" 
                    for vChar in vCheckWord:
                        if vChar not in vPunctuations:
                            vNoPunctuations = vNoPunctuations + vChar
                
                    vNoPunctuations = vNoPunctuations.lower()
                    
                    vStart_Index = str(vNoPunctuations).find(vWord)
                    vEnd_Index = vStart_Index + len(vWord) # if the start_index is not -1
                    vStringLength = len(str(vNoPunctuations))

                    # Check if vWord exists as the 1st Word in String
                    if vStart_Index == 0:
                        vWord = vWord + " "
                    elif vEnd_Index == vStringLength:
                        vWord = " " + vWord
                    elif vStart_Index > 1:
                        vWord = " " + vWord + " "
                
                    if str(vWord) in str(vNoPunctuations):
                        vFinalCategory[i] == "Body_Systems"
                        vTempArray = list(vFeaturesDictionary["skin"])
                        vTempArray[i] = 1
                        vFeaturesDictionary["skin"] = vTempArray
                    
                    #--------------------------------------------------------------
                    vWord = str(vBodySystemArray[w])
                    vCheckWord = str(vCaseString)
                    vPunctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''#This code defines punctuation
                    # This code removes the punctuation
                
                    vNoPunctuations = "" 
                    for vChar in vCheckWord:
                        if vChar not in vPunctuations:
                            vNoPunctuations = vNoPunctuations + vChar
                
                    vNoPunctuations = vNoPunctuations.lower()
                    
                    vStart_Index = str(vNoPunctuations).find(vWord)
                    vEnd_Index = vStart_Index + len(vWord) # if the start_index is not -1
                    vStringLength = len(str(vNoPunctuations))
                
                    #print(" vNoPunctuations ", vNoPunctuations, " str(l) ", str(l), " vStart_Index ",vStart_Index," vEnd_Index ", vEnd_Index," vStringLength ",vStringLength)
                
                    # Check if vWord exists as the 1st Word in String
                    if vStart_Index == 0:
                        vWord = vWord + " "
                    elif vEnd_Index == vStringLength:
                        vWord = " " + vWord
                    elif vStart_Index > 1:
                        vWord = " " + vWord + " "
                
                    if str(vWord) in str(vNoPunctuations):
                        vFinalCategory[i] == "Body_Systems"
                        vTempArray = list(vFeaturesDictionary["skin"])
                        vTempArray[i] = 1
                        vFeaturesDictionary["skin"] = vTempArray
                
    #--------------------------------------------------------------------------
    # Output csv with features and scores, final category and accident cases
    # Output a list of features as csv column
    
    vOutputHeader = []
    
    for i in vFeaturesDictionary:
        vOutputHeader.append(i)
        vCasesDataFrame[i] = list(vFeaturesDictionary[i])
    
    vCasesDataFrame[vCategory] = vFinalCategory
    vOutputHeader.append(vCategory)
    vOutputHeader.append("Case_Title")
    vOutputHeader.append("Cases")
    vCasesDataFrame.to_csv(vOutput, columns = vOutputHeader)

    return vFeaturesDictionary

# -----------------------------------------------------------------------------
# Semantic Relationships
# Function to populate categories based on sementic relations of a word
# -----------------------------------------------------------------------------

# Functions to find substring between two words of a string
def fFind_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def fFind_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def fWordSemantics(vLoadCSV_Dictionary, vOutput, vOutput1):
    
    #--------------------------------------------------------------------------
    # Load Dictionary
    vDictionaryDataFrame = pd.read_csv(vLoadCSV_Dictionary, encoding="ISO-8859-1")
    
    #print(vDictionaryDataFrame["Case_Title"])
    
    #--------------------------------------------------------------------------
    # Clean and process
    vCaseDictionary = {}
    vCaseListCombo = {}
    vCaseCount = 0
    
    for i in vDictionaryDataFrame["Cases"]:
        # Convert the Free Text into Tokens
        vWordTokens = word_tokenize(str(i))
        
        # Remove Punctuations
        vWordTokens_nop = [ t for t in vWordTokens if t not in string.punctuation]
        
        # Convert All Lower Case
        vWordTokens_lower=[ t.lower() for t in vWordTokens_nop]
        
        # Create Stopwords
        vStopWords = stopwords.words('english')
        
        vPunctuationsLen = len(gPunctuations)
        
        for iPunctuations in range (vPunctuationsLen):
            vStopWords.append(gPunctuations[iPunctuations])
            
        vUpdateStopLen = len(gUpdateStopwords)
        for iUpdateStopLen in range(vUpdateStopLen):
            vStopWords.append(gUpdateStopwords[iUpdateStopLen])
        
        # Remove All Stopwords From Text
        vWordTokens_nostop=[ t for t in vWordTokens_lower if t not in vStopWords]
        
        # Pos Tag All Tokens
        vPosTagWordTokens = nltk.pos_tag(vWordTokens_nostop)
        
        # Identify VBG tags <actions> NN
        # Rule 1 - Find VBG -> NN
            # If VBG ends with Fullstop or end of string before finding NN
            # Rule 2 - Find NN <- VBG
            # Work backwards to find NN before VBG
        
        vCurrentCase = {}
        vTempListCombo = []
        
        # Get all VBG tags in tokens
        for j in range(len(vPosTagWordTokens)):
            vTempCompleteSubString = ""
            vTempVBG_NNTokenPair = ""
   
            if vPosTagWordTokens[j][1] == "VBG":
                #vTempVBGTokens.append(vPosTagWordTokens[j][0])
                
                # Find the next NN after this VBG
                vCount = 0
                vMax = 10 # Rule of Thumb
                vSubstring = ""
                
                for k in range(vMax):
                    if (j+k) < len(vPosTagWordTokens): # make sure doesn't exceed index
                        if vPosTagWordTokens[j+k][1] == "NN":
                            # Find substring between VBG and NN. Is there fullstop?
                            vSubstring = fFind_between(i, vPosTagWordTokens[j][0], vPosTagWordTokens[j+k][0])
                            
                            if "." in vSubstring:
                                # If there is a fullstop
                                for l in range(vMax):
                                    vReverseIndex = j-l
                                    if vReverseIndex < 0: # make sure index is not less than 0
                                        vReverseIndex = 0
                                    
                                    # Find NN by going backwards from VBG
                                    if vPosTagWordTokens[vReverseIndex][1] == "NN":
                                        # Find substring between VBG and NN. Is there fullstop?
                                        vSubstring1 = fFind_between_r(i, vPosTagWordTokens[vReverseIndex][0], vPosTagWordTokens[j][0])
                                        if "." in vSubstring:continue
                                        # If there is a fullstop
                                        # Discard this round of VBG
                                        else:
                                            vTempVBG_NNTokenPair = str(vPosTagWordTokens[vReverseIndex][0])+"_"+str(vPosTagWordTokens[j][0])
                                            vTempCompleteSubString = str(vPosTagWordTokens[vReverseIndex][0] + vSubstring1 + vPosTagWordTokens[j][0])
                                            vCurrentCase[vTempVBG_NNTokenPair] = vTempCompleteSubString
                                            vTempListCombo.append(vTempVBG_NNTokenPair)
                                        l = vMax
                            else:
                                vTempVBG_NNTokenPair = str(vPosTagWordTokens[j][0]) +"_"+ str(vPosTagWordTokens[j+k][0])
                                vTempCompleteSubString = str(vPosTagWordTokens[j][0] +" "+ vSubstring +" "+ vPosTagWordTokens[j+k][0])
                                vCurrentCase[vTempVBG_NNTokenPair] = vTempCompleteSubString
                                vTempListCombo.append(vTempVBG_NNTokenPair)
                                
                            #print(vTempCompleteSubString)
                            #vTempNNTokens.append(vPosTagWordTokens[j+k][0])
                            k = vMax
        vCaseDictionary[str(vCaseCount)] = vCurrentCase
        vCaseListCombo[str(vCaseCount)] = vTempListCombo
        #print(vCurrentCase)
        vCaseCount += 1
        #print(vCaseDictionary)

    # Output to CSV
    
    vOutputHeader = []
    
    for i in vDictionaryDataFrame:
        vOutputHeader.append(i)
    
    vFinal1 = []
    for i in vCaseListCombo:
        vFinal1.append(str(vCaseListCombo[i]))
    
    vFinal2 = []
    for i in vCaseDictionary:
        vTempArray = []
        for j in vCaseDictionary[i]:
            vTempString = str(vCaseDictionary[i][j])
            vTempArray.append(vTempString)
        
        vTempArrayString = ', '.join(vTempArray)
        vFinal2.append(str(vTempArrayString))
    
    #vDictionaryDataFrame["Action and Object"] = vFinal1
    #vDictionaryDataFrame["Common Activities"] = vFinal2
    #vOutputHeader.append("Action and Object")
    #vOutputHeader.append("Common Activities")
    #vDictionaryDataFrame.to_csv(vOutput, columns = vOutputHeader)
    
    f = open(vOutput, 'w', encoding="ISO-8859-1", newline="\n") # open a csv file for writing
    f.write("Action and Object" + ',' + "Common Activities" + "\n")
    
    for vM in range(len(vFinal2)):
        # --- Write CSV headers
        vTempString = vFinal2[vM]
        vTempString.replace(",", "_")
        
        f.write(str(vFinal1[vM]) + ',' + vTempString + "\n")
    
    f.close() 
    
    
    # Output row by row of common activities for analyse
    f = open(vOutput1, 'w', encoding="ISO-8859-1", newline="\n") # open a csv file for writing
    f.write("Action and Object" + ',' + "Common Activities" + "\n")
    
    for i in vCaseDictionary:
        for j in vCaseDictionary[i]:
            f.write(str(j) + ',' + str(vCaseDictionary[i][j]) + "\n")
    
    f.close() 
            
            
            
            
            
            
            
    return vCaseDictionary

# -----------------------------------------------------------------------------
# Machine Learning Functions

# -----------------------------------------------------------------------------
# ML01 - SVC
# -----------------------------------------------------------------------------

def gML_SVC01(vLoad_Features, vLoadCSV_Train, vLoadCSV_Test, vSaveModel):
    
    #--------------------------------------------------------------------------
    # vLoad_Features contains and array of feature names
    # 1st feature is the predict target for train and test
    # Get only the predictor labels
    
    vLabelsArray = []
    
    vFeaturesLength = len(vLoad_Features)
    
    for i in range(vFeaturesLength):
        # Leave out the 1st feature
        if i > 0:
            vLabelsArray.append(vLoad_Features[i])
    
    
    #--------------------------------------------------------------------------
    # Load csv for training model
    # Feed into training model
    
    vTrainDataframe = pd.read_csv(vLoadCSV_Train, header = None, names = vLoad_Features)
    
    vTrainLabels = vTrainDataframe.Cause
    vLabels = list(set(vTrainLabels))
    vTrainLabels = np.array([vLabels.index(x) for x in vTrainLabels])
    vTrainFeatures = vTrainDataframe.iloc[:,1:]
    vTrainFeatures = np.array(vTrainFeatures)  
    
    # vClassifier = svm.SVC(probability=True, C=25)
    vRandomSeed = 2017
    vClassifier = svm.SVC(C = 10.0, gamma = 0.01, random_state = vRandomSeed)
    vClassifier.fit(vTrainFeatures, vTrainLabels)
    
    #--------------------------------------------------------------------------
    # Load csv to test model
    # Feed into model
    
    vTestDataframe = pd.read_csv(vLoadCSV_Test, header = None, names = vLoad_Features)
    
    vTestLabels = vTestDataframe.Cause
    vLabels = list(set(vTestLabels))
    vTestLabels = np.array([vLabels.index(x) for x in vTestLabels])

    vTestFeatures = vTestDataframe.iloc[:,1:]
    vTestFeatures = np.array(vTestFeatures)
    
    #--------------------------------------------------------------------------
    # Output Test Data Predictions
    # Save Prediction in CSV
    
    vResults = vClassifier.predict(vTestFeatures)
    #print("Results :", vResults)
    
    vResultString = ""
    vResultsLength = len(vResults)
    
    for i in range(vResultsLength):
        vResultString += str(vResults[i]) + ","
        
    #print("vResultString :",vResultString)
    
    vFile = open("ML01_TestPredict.csv", 'w', newline="\n")
    vFile.write(str(vResultString) + "\n")
    vFile.close
    
    vNumCorrect = (vResults == vTestLabels).sum()
    vRecall = vNumCorrect / len(vTestLabels)
    print("num_correct :", vNumCorrect)
    print("model accuracy (%) :", vRecall * 100, "%")
    
    #--------------------------------------------------------------------------
    # Save model
    vFilename = vSaveModel + "Finalised_ML01_Model.sav"
    pickle.dump(vClassifier, open(vFilename, 'wb'))
    
    return


def gML_Pred01(vLoad_Features, vLoadCSV_Pred, vOutputCSV_Pred, vLoadModel):
    
    #--------------------------------------------------------------------------
    # Load ml model
    # Load CSV for prediction
    # Feed data into ml model
    vLoadedModel = pickle.load(open(vLoadModel, 'rb'))
    
    #--------------------------------------------------------------------------
    # Load csv to test model
    # First column of csv is empty
    # Feed into model
    
    vTestDataframe = pd.read_csv(vLoadCSV_Pred, header = None, names = vLoad_Features)

    vTestFeatures = vTestDataframe.iloc[:,1:]
    vTestFeatures = np.array(vTestFeatures)
    
    #--------------------------------------------------------------------------
    # Output Test Data Predictions
    # Save Prediction in CSV
    vResults = vLoadedModel.predict(vTestFeatures)
    print(vResults)
    
    vResultString = ""
    vResultsLength = len(vResults)
    
    for i in range(vResultsLength):
        vResultString += str(vResults[i]) + ","
        
    print("vResultString :", vResultString)
    
    vFile = open("ML01_RealWorld_Predict.csv", 'w', newline="\n")
    vFile.write(str(vResultString) + "\n")
    vFile.close
    
    return


# -----------------------------------------------------------------------------
# Initialisation Dictionary and Association Rules
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Events Causing Accidents" Dictionary
vLoad_Features = ["Contact_With_Objects_Vechicles_Or_Equipment","Exposure_To_Harmful_Substances_Or_Environments","Falls_Slips_Trips","Fires_And_Explosions","Overexertion_And_Bodily_Reaction","Violence_Inflicted_Harm_By_People_Or_Animals"]
vLoadCSV_Dictionary = "Library/1_BoW_EventsCauses.csv"
gEventCauseDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Events Causing Accidents" Categories
gAssociation2Category("Events_Causes", gEventCauseDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withEventsCauses.csv")

#gAssociation2Category("Events_Causes", gEventCauseDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withEventsCauses.csv")
gAssociation2Category("Events_Causes", gEventCauseDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withEventsCauses.csv")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Fatal or Survived" Dictionary
vLoad_Features = ["Fatal"]
vLoadCSV_Dictionary = "Library/1_BoW_Fatality.csv"
gFatalityDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Fatal or Survived" Categories
gAssociation2Category("Fatal", gFatalityDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withFatality.csv")

#gAssociation2Category("Fatal", gFatalityDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withFatality.csv")
gAssociation2Category("Fatal", gFatalityDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withFatality.csv")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Construction and Building Industry" Dictionary
vLoad_Features = ["Construction and Building Industry"]
vLoadCSV_Dictionary = "Library/1_BoW_Construction.csv"
gConstructionAndBuildingDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Construction and Building Industry" Categories
gAssociation2Category("Construction_and_Building", gConstructionAndBuildingDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withConstructionBuilding.csv")

#gAssociation2Category("Construction_and_Building", gConstructionAndBuildingDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withConstructionBuilding.csv")
gAssociation2Category("Construction_and_Building", gConstructionAndBuildingDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withConstructionBuilding.csv")
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Load "Occupations" Dictionary
vLoad_Features = ["Construction Supervisor/Safety Officer/Construction Worker or Laborer","Infrastructure Mechanic/Landscape Engineer/Building Engineer/Building Designer","Civil Engineer","Machine Operator/Driver"]
vLoadCSV_Dictionary = "Library/1_BoW_Occupation.csv"
gOccupationsDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Occupations" Categories
gAssociation2Category("Occupation", gOccupationsDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_witOccupation.csv")

#gAssociation2Category("Occupation", gOccupationsDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_witOccupation.csv")
gAssociation2Category("Occupation", gOccupationsDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withOccupation.csv")

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Body Parts" Dictionary
vLoad_Features = ["Head_and_Throat","Upper_Extremities_and_Trunk","Lower_Extremities_and_Trunk","Body_Systems"]
vLoadCSV_Dictionary = "Library/1_BoW_BodyParts.csv"
gBodyPartsDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Body Parts" Categories
gAssociation2Category("Parts_Of_Body", gBodyPartsDictionary, "BP", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withBodyParts.csv")

#gAssociation2Category("Parts_Of_Body", gBodyPartsDictionary, "BP", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withBodyParts.csv")
gAssociation2Category("Parts_Of_Body", gBodyPartsDictionary, "BP", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withBodyParts.csv")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Actions" Dictionary
vLoad_Features = ["working","transporting","falling","lifting","pressing","operating","releasing","cutting","wearing","rotating","drilling","inhaling","excavating","driving","walking","landing","rigging","parking","preparing","climbing","ejecting","grinding","knocking","unloading","rolling","burying","servicing","roofing","traveling","riding","manufacturing","throwing","erecting","filling","breathing","welding","adjusting","molding","footing","cycling","packing","milling","irrigating","lowering","conducting","bearing","picking","feeding","casing","knocking","resting","inserting","paving","shipping","cooking","tailgating","sandblasting","harvesting","refueling","towing","disengaging","mowing","swinging","swimming","vibrating","slicing","bonding","kneeling","maintaining","functioning","drinking","speeding","bolting","skidding","ironing","supervising","fabricating","walking","ripping","dragging","shifting","tapping","tooling","screwing","repairing","detaching","carrying"]
vLoadCSV_Dictionary = "Library/1_BoW_Actions.csv"
gActionsDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

vActionDict_Malaysia = {}
vActionDict_OSHA = {}

# Populate "Actions" Categories
vActionDict_Malaysia = gAssociation2Category("Actions", gActionsDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withActions.csv")

#vActionDict_Malaysia = gAssociation2Category("Actions", gActionsDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withActions.csv")
vActionDict_OSHA = gAssociation2Category("Actions", gActionsDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withActions.csv")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load "Objects" Dictionary
vLoad_Features = ["truck","machine","coworker","hospital","ladder","tank","metal","tree","pipe","concrete","steel","door","coworkers","aircraft","horse","bull","kettle","phone","crane","equipment","saw","trailer","forklift","conveyor","system","tractor","car","hydraulic","tool","rig","loader","multiple","cab","grinder","mixer","oven","mech","mechanism","machinery","compressor","wrench","knife","bus","helicopter","tanker","scissor","compactor","chipper","automobile","railcar","piston","refrigerator","screwdriver","cooker","gunshot","turbine","plier","scooter","iron","plywood","cement","construction","center","facility","industrial","department","dock","rail","bridge","highway","bed","vessel","railroad","residential","agriculture","mill","memorial","city","boat","residence","community","university","clinic","francisco","garage","basement","port","ship","apartment","lawn","jobsite","stage","restaurant","outlet","walkway","worksite","airport","gravel","hotel","pier","drywall","driveway","sawmill","foundry","runway","complex","dam","chimney","hallway","shipyard","patio","skyline","vineyard","nursery","sideways","bakery","incinerator","doorway","undrgrd","jet","garden","theater","backyard","condominium","classroom","southeast","corridor","prison","texas","college","washington"]
vLoadCSV_Dictionary = "Library/1_BoW_Objects.csv"
gObjectsDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

vObjectsDict_Malaysia = {}
vObjectsDict_OSHA = {}

# Populate "Objects" Categories
vObjectsDict_Malaysia = gAssociation2Category("Objects", gObjectsDictionary, "Y", "Input_Output/MalaysiaCases_AddOns.csv", "Input_Output/MalaysiaCases_withObjects.csv")

#vObjectsDict_Malaysia = gAssociation2Category("Objects", gObjectsDictionary, "Y", "Input_Output/MalaysiaCases_default.csv", "Input_Output/MalaysiaCases_withObjects.csv")
vObjectsDict_OSHA = gAssociation2Category("Objects", gObjectsDictionary, "Y", "Input_Output/OshaCases_default.csv", "Input_Output/OshaCases_withObjects.csv")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load Semantic POSTAGGING function
vLoadCSV_Dictionary = "Input_Output/OshaCases_default.csv"
vOutput = "Input_Output/Cases_of_Common_Activities.csv"
vOutput1 = "Input_Output/Cases_of_Common_Activities_ONLY.csv"
gSemanticCommonActivities = fWordSemantics(vLoadCSV_Dictionary, vOutput, vOutput1)

# -----------------------------------------------------------------------------
# Load "Activities" Dictionary
# To use semantic relationships to find the activity accompany action words
#vLoad_Features = [""]
#vLoadCSV_Dictionary = "LoremIpsum.csv"
#gActivitiesDictionary = gLoadDictionary(vLoad_Features, vLoadCSV_Dictionary)

# Populate "Activities" Categories
#gAssociation2Categories("Activities", gActivitiesDictionary)
# -----------------------------------------------------------------------------
vLoad_Features = ["Cause","abrasion","abrasive","bruising","concussion","contusion","cut","elec","electric","electrical","electrocuted","electronic","fractured","fracturing","lacerated","lacerates","lacerating","laceration","severed","shock","struck","wound","transportation vehicles","transportation","powered industrial vehicles","powered industrial","powered mobile industrial equipment","powered mobile","industrial equipment","vehicle","mobile equipment","traffic","loss of control","loss control","sudden stop","sudden start","jolting of","pedestrians","roadway","roadway workers","struck by vehicle","motion of","means of transportation","means of transport","occupant of","operated vehicles","computer","subways","subway","monorails","monorail","roadways","highway","street","travel","aircraft","rail vehicle","pedestrian vehicular","motorized land vehicle","motorized vehicle","motorised land vehicle","motorised vehicle","vehicle collision","crashes","crash","highway vehicles","motorized","motorised","autos","buses","trucks","motorcycles","RVs","vehicles?powered","airplanes","gliders","parachutes","trains","amusement park rail vehicles","amusement park rail vehicle","amusement park rail","computerized airport trams","computerized airport tram","computerised airport trams","computerised airport tram","fishing boats","fishing boat","sailboats","sailboat","canoes","canoe","ATVs","ATV","golf carts","golf cart","snowmobiles","snowmobile","segways","segway","tractors","tractor","mobile construction","loaders","loader","bulldozers","bulldozer","backhoes","backhoe","mobile cranes","mobile crane","cranes","crane","skidders","skidder","harvesters","harvester","mobile planters","mobile planter","planters","planter","human powered vehicles","human powered vehicle","ridden","rodeo horses","rodeo horse","drawn wagons","horse","wagons","wagon","bicycles","bicycle","military vehicles","military vehicle","collision","derailment","locomotive strikes","locomotive","locomotive strike","knifed","overturned","bump","struck by","run over","over by","caught between","transport","rolling","roll","propeller blades","propeller blade","rolled over","swinging buckets","swinging booms","swinging doors","swinging bucket","swinging boom","swinging door","falling truck beds","falling forks","forklifts","forklift","falling object","crushed","crush","pinned","pin","caught under","bouncing","struck by discharged","flying object","thrown","hurled","propelled","flies off","breaks off","striking","flew off","ejected","swinging","bumping into","bump into","caught in","compressed by","compressed","compress","entangle","entangled","engulfment","landslides","cave","collapsing structures","collasp","crane collapses","crane collapse","falling debris","structure collapse","trenching","avalanche","collapsing grain","collapsing palm fronds","rubbed","abraded","friction","blisters","scratches","abrasions","amputate finger","amputate toe","amputate leg","amputate hand","amputate arm","hit by","bleeding","electrocution","collapsed","crushed by","backhoe-like","swing saddle/a-frame","swing saddle","a-frame","aquamog","challenger","lift model","amputation","fan belt","auger","substance","substances","harmful environment","harmful gas","bad air","electricity","lightning","electric shock","electric shocks","power source","live wire","electrical arc","electrified","electric fences","electric fence","voltage","high voltage","volts","power lines","power line","industrial transformers","industrial transformer","arc flashes","arc flash","electrified piping","live power lines","live power line","radiation","ionizing","ionising","ultraviolet","laser light","laser","infrared light","infrared","welding flash","flash burns","sunburn","sun poisoning","eye injuries","eye injury","laser beams","laser beam","electrical sparks","electrical spark","microwaves","microwave","radio waves","radio wave","radars","radar","power frequencies","power frequency","radiant heat burns","heat burns","heat burn","hot objects","hot object","hot substances","hot substance","sunstroke","environmental heat","heat","hearing impairments","hearing","hearing impairment","loud noise","loud noises","prolonged noise","hearing loss","brief exposure to noise","prolonged exposure to noise","repeated exposure to noise","exposure to noise","exposure to temperature extremes","temperature extreme","exposure to heat","exposure to cold","heat exhaustion","heat stroke","freezing","frostbite","hypothermia","welding torches","welding torch","heated fluids","dry ice","freezer surfaces","liquid nitrogen","air pressure","water pressure","pressure","pressure change","sea diving","deep","aircraft decompression","decompression","aircraft altitude","pressurized air","pressurised air","air discharged","struck by discharge","inhalation","absorption","skin contact","injection","needlestick","ingestion","swallowing","harmful substances","allergic reactions","allergic","allergy","allergen","contagious","infectious","diseases","drug overdoses","exposure to","caustic","noxious","allergenic","infectious agents","infectious agent","manufacture of","dispensing  of","administration of","therapeutic","vaccines","piercing of the skin","piercing of skin","temperature extremes","infectious sharps","scalpels","tubing","medicines","fatal drug","blood or body fluid","body fluid","body fluid splash","dermatitis","chemical burns","skin absorption","lyme disease","west nile virus","scabies","rabies","oxygen deficiency","lack of oxygen","submersions","sewer gas","depletion of oxygen","oxygen deficient","restriction of breathing","asphyxiated","fumes","toxic","suffocate","gas","engulfed","chemical","exposure","disease","contracted","infected","bacteria","cave-in","drowning","drown","drowned","bottom of the pool","buried","while diving","given CPR","insecticide","poisoning","poison","confined space","exposed to","spray paint","anhydrous ammonia vapor","vapor ","ammonia vapor ","anhydrous","ammonia","hydrogen sulfide overexposure","hydrogen sulfide","overexposure","airborne","felling","sprain","tipped","falls","fell","falling","fall","slips","slip","slipped","tripped","tripping","trips","trip","jumping","jump","jumps","jumped","slipping","slippery","slippery surface","slippery surfaces","uneven","force of impact","burn","burned","flame","ignited","ignites","smoke","explosion","fire","burning building","burning","inhaled","trapped in a fire","trapped in fire","oxygen","heat source","stove tops","stove top","ovens","oven","burners","burner","grills","hot","catch fire","demolition","blasting explosion","implode","buildings","blast","dynamite","mining explosions","ignition of","ignite","forest fire","bush fire","explode","implosion","coal explosion","exploding","exploded","stroke","suffocated","overexertion","bodily reaction","bodily motion","excessive physical","unnatural position","microtasks","repetitive lifting of","lifting of","trashcans","files","luggage","trays","lifting furniture","lifting crates","lifting crate","lifting construction materials","shaking out","healthcare","clerical","typing","key entry","texting","mousing","musical","medical","scanning","skinning","assembly","signing","prolonged","calisthenics","ups","choking with","natural cause","natural causes","cardiac arrest","fainting","faint","cardiac","illness","overexert","ill","unwell","preexisting condition","alcohol abuse","intentional injuries","intentional injury","weapons","firearms","firearm","stun guns","stun gun","direct physical","physical contact","inflicted by","inflicted","shot","charged gun","handgun","shotgun","rifle","paintball guns","paintball gun","bows","bow","BB guns","BB gun","hitting","kicking","beating","clubbing","bludgeoning","hitting with weapon","fighting","grabbing","grappling","biting","pushing","pinching","squeezing","shoving","strangulation","bombing","arson","rape","sexual assault","assault","threat","verbal assault","violent acts","violent act","hanging","self","animal","insect","stings","venom","venomous bites","venomous","bees","wasps","hornets","yellow jackets","sea nettles","jelly fish","spider bites","scorpion bites","fire ant bites","fire ant stings","venomous snake","venomous snake bites","rattlesnakes","rattlesnake","copperheads","copperhead","cottonmouths","cottonmouth","water moccasins","water moccasin","kicked by","mauled","clawed","scratched","gored","violence","inflict","suicide","killed","died","dead","die","death","kill","deadman","fatal","fatality","drowned","drown","drowning","asphyxiated","dies","asphyxiate","deceased","electrocution","electrocuted","sky lift overturned","crushed","buried","decapitated","decapitate","head","face","neck","eye","skull","nose","jaw","throat","ear","forehead","facial","hair","mouth","teeth","tongue","tooth","cranial","brain","scalp","ears","eyes","nasal","nasal cavity","internal nasal","nasopharynx","nasal passage","sinus","cheek","chin","lips","gum","vocal cord","vocal cords","larynx","laryngopharynx","pharynx ","trachea","concussion","body","chest","thumb","rib","bone","nail","skin","vertebra","torso","limb","tendon","ligament","abdominal","right-hand","spine","spinal cord","thoracic","lumbar","sacral","coccygeal","abdomen","stomach","spleen","urinary","bladder","kidney","kidneys","intestines","peritoneum","colon","rectum","liver","gallbladder","pancreas","upper extremities","shoulder","shoulders","clavicle","clavicles","scapula","scapulae","arm","arms","upper arm","upper arms","elbow","elbows","forearm","forearms","wrist","wrists","hand","hands","finger","fingers","fingernail","fingernails","nailbed","nailbeds","fingertip","fingertips","femur","waist","fibula","hip","hips","pelvis","buttock","buttoks","external reproductive","reproductive tract","scrotum","penis","prostate","testis","testes","ovary","ovaries","uterus","genital","female genitals","female genital","lower extremities","leg","legs","thigh","thighs","knee","knees ","lower leg","lower legs","ankle","ankles","foot","feet","sole","soles","ball of the foot","balls of the foot","arch","arches","instep","insteps","heel","heels","toes","toe","toenail","toenails","butt","groin","pelvic","ass","calf","heart","lung","pulmonary","artery","cardiopulmonary","construction worker","construction","manual labor","laborer","laboring","hard hat","concrete","shotcrete","gunite","grouting","steel forms","demolition","concrete cutting","pavement breaking","cutting torch","environmental remediation","hazardous waste","fences","fence","landscaping","street sweeping","hod carrier","paving","white paving formwork","traffic control","striping","signs","piping","water pipe","sewer and storm drain","gravedigger","tunnels","drilling","blasting","mason","manual laborer","construction supervisor","supervise","supervising","site manager","manager","construction manager","building manager","site agent","oversee operations","oversee operation","oversee","quality standards","quality standard","standard","standards","safety officer","safety","ensure safety","inspect","health and safety","EHS","CSR","EICC","work procedure","safety work","risk","risk assessment","safety management","ISO 14001","ISO","14001","OSHA 18001","OSHA","18001","MOM","NEA","PUB","WSH","inspection","on-site","on site","site","roof mechanic","roof construction","waterproof","weatherproof building","weatherproof buildings","substrate","installed on","rafters","beams","trusses","carpentry","cladding","roof cladding","asphalt shingles","3-tab","architectural","dimensional","tile","tiles","concrete tiles","clay tiles","single-ply","EPDM","rubber","PVC","TPO","rubber shingles","metal panels","wood shakes","liquid-applied","hot asphalt","hot rubber","foam","thatch","solar tile","solar tiles","duro-last","living roof","rooftop landscape","rooftop landscapes","roofer","roofers","carpert layering","carpert laying","floor layer","floor installer","vinyl laying","timber laying","furniture","landscaper","painter","painting","decorator","decorating","plumber","pipefitter","designer","specialty roofs","infrastructure","mechanic","engineer","landscape","builder","civil engineer","civil","design","maintenance","road","roads","canal","canals","dam","dams","public sector","coastal engineering","coastal engineer","control engineering","control engineer","earthquake engineering","earthquake engineer","environmental engineering","environmental engineer","forensic engineering","forensic engineer","geotechnical engineering","geotechnical engineer","plant engineering","plant engineer","structural engineering","structural engineer","surveying","surveyor","transportation engineering","transportation engineer","municipal engineering","municipal engineer","urban engineering","urban engineer","urban planner","water resources engineering","water resources engineer","water resource","water resources","planner","infrastructures","electrician","electrical","wire","elctricity","millwright","corn mills","corn mill","mill","mill machinery","plasterer","plastering","plaster","layer of plaster","interior wall","decorative molding","decorative moldings","plasterwork","house builder","forklift","plans","instruct","boilermaker","steel fabrication","steel","steel fabrications","plates","tubes","boiler","bridge","bridges","blast furnace","blast furnaces","mining equipment","boilersmith","shipbuilding","engineering","shipyard construction","shipyard constructing","iron boiler","iron boilers","boilermaking","welding","fitting tube","fitting tubes","power plant","power plants","stress fracture","stress fractures","rust","corrosion","high steam pressure","high steam pressures","re-fitting boiler","re-fitting a boiler","seagoing","seagoing vessel","remodeling steam plant","remodeling of a steam plant","boiler repair","domestic boiler","domestic boilers","re-tubing","hot water boiler","hot water boilers","pressure vessels","pressures","oxy-acetylene","gas torch","gouge steel","gas tungsten","arc welding","GTAW","shielded metal arc welding","metal arc","SMAW","gas metal arc welding","GMAW","r stamp welding","r stamp","powerpiping","ironworker","sheet metal worker","sheet metal","welder","driver","driving","forklift driver","class 3","class 4","lorry crane driver","lorry","class 5","track-type","agricultural tractor","agricultural tractors","bulldozer","snowcat","track skidder","skidder","tractor","vehicle","military engineering vehicle","military engineering vehicles","grader","skidsteer","loader","amphibious excavator","compact excavator","dragline excavator","dredging","bucket-wheel excavator","excavator","long reach excavator","reclaimer","suction excavator","walking excavator","trencher","yarder","feller buncher","harvester","track harvester","wheel forwarder","wheel skidder","pipelayer","sideboom","scraper","fresno scraper","wheel tractor-scraper","articulated hauler","hauler","articulated truck","truck","compactor","wheel dozers","soil compactors","soil stabilizer","skip loader","skippy","wheel loader","front loader","integrated tool carrier","track Loader","material Handler","aerial work platform","lift table","cherry picker","crane","knuckleboom loader","trailer mount","straddle carrier","reach stacker","telescopic handlers","asphalt paver","asphalt plant","cold planer","cure rig","paver","pavement milling","pneumatic tire compactor","roller","road roller","roller compactor","slipform paver","vibratory compactor","roadheader","tunnel boring","tunnel boring machine","ballast tamper","drilling machine","rotary tiller","rototiller","rotovator","dump truck","transit-mixer","lowboy","trailer","street sweeper","dredger","elevator mechanic","elevator","heavy equipment operator","heavy equipment","operator","pile driver","pile","machine"]
vLoadCSV_Train = "Train_Test_Pred_DataSets/1_Round2/TrainData.csv"
vLoadCSV_Test = "Train_Test_Pred_DataSets/1_Round2/TestData.csv"
vSaveModel = "Train_Test_Pred_DataSets/1_Round2/Prediction/"
gML_SVC01(vLoad_Features, vLoadCSV_Train, vLoadCSV_Test, vSaveModel)

vLoadCSV_Pred = "Train_Test_Pred_DataSets/1_Round2/OSHAData.csv"
vOutputCSV_Pred = "Train_Test_Pred_DataSets/1_Round2/OshaOutput.csv"
vLoadModel = "Train_Test_Pred_DataSets/1_Round2/Prediction/Finalised_ML01_Model.sav"
gML_Pred01(vLoad_Features, vLoadCSV_Pred, vOutputCSV_Pred, vLoadModel)