import csv
import math
import statistics as stat
import numpy as np
#import tensorflow as tf

def readTxtFile(txtfile, intornot):
    file  = open(txtfile, 'r')
    List = []
    for line in file:
        if(intornot):
            List.append(float(line[:-2]))
        else:
            List.append(line[:-1])
    return(List)

def OpenCsvFile (csvfile):
    
   f=open(csvfile,'r') # opens file for reading
   reader = csv.reader(f, delimiter=',')
   Name,Valence_mean,Valence_SD,Valence_N,Arousal_mean,Arousal_SD,Arousal_N = [],[],[],[],[],[],[]
   next(reader)
   for row in reader:
      Name.append(row[1])
      Valence_mean.append(float(row[4]))
      Valence_SD.append(float(row[5]))
      Valence_N.append(float(row[6]))
      Arousal_mean.append(float(row[7]))
      Arousal_SD.append(float(row[8]))
      Arousal_N.append(float(row[9]))
   f.close()
   return (Name,Valence_mean,Valence_SD,Valence_N,Arousal_mean,Arousal_SD,Arousal_N)


def OpenCsvFile_Gaped(csvfile):
    f= open(csvfile,'r')
    reader = csv.reader(f, delimiter=',')
    Name,Valence_mean,Arousal_mean = [],[],[]
    next(reader)
    for row in reader:
        Name.append(row[0])
        Valence_mean.append(float(row[1]))
        Arousal_mean.append(float(row[2]))
    f.close()
    return(Name,Valence_mean,Arousal_mean)

def accuracy(predictions, labels):
      sumWeight = 0
# =============================================================================
#       levelName = ['Moderately_Low','Somewhat_low','Neither_Low_nor_High','Somewhat_High','Moderately_High']
#       LevelEquivalent = [1,2,3,4,5]
# =============================================================================
      for j in range(predictions.shape[0]):
          sumWeight = sumWeight+ (abs(abs(labels[j]+1-float(predictions[j]+1))-4.)/4.) #weight system: 0;0.25;0.5;0.75;1
      return (100.0 * float(sumWeight) / predictions.shape[0])
  
def RMSE(List):
    sumF = 0
    n = List[1].shape[0]
    for i in range(n):
        sumF = sumF + (int(List[1])-int(List[0]))**2
    return(np.ndarray(math.sqrt(sumF/n)))
    
def PearsonCC(predictions,labels):
#    predictions = normalize(predictions)
#    labels = normalize(labels)
    '''computes Pearsonâ€™s Correlation Coefficient (CC)'''
    return( np.cov(predictions,labels)[0][1]/(np.std(predictions)*np.std(labels)))

def ConcordanceCC(predictions,labels):
    '''computes Concordance Correlation Coefficient'''
#    predictions = normalize(predictions)
#    labels = normalize(labels)
    return(2*PearsonCC(predictions,labels)*np.std(predictions)*np.std(labels)/ \
           (stat.variance(predictions)+stat.variance(labels)+ (np.mean(predictions)- np.mean(labels))**2))

def SAGR(predictions,labels):
    '''Computes Sign Agreement Metric (SAGR)'''
    sumF = 0
    n = len(predictions)
    for i in range(n):
        if(np.sign(predictions[i])==np.sign(labels[i])):
            #if prediction/labels == 0, sign return 0
            sumF += 1
    return(sumF/n)


def normalize(listN):
    '''Performs a min-max normalization'''
    # normalizes between 0 and 1
    maxList = max(listN)
    minList = min(listN)
    for i in range(len(listN)):
        listN[i] = (listN[i]-minList)/(maxList - minList)
    return listN

def rescaling(listN,currentmin,currentmax):
    '''Rescales the List around 0'''
    neutral = (currentmax+currentmin)/2
    for i in range(len(listN)):
        listN[i] = listN[i]-neutral
    return listN   
    
def normalizerange(listN,newmin,newmax):
    normalize(listN)# normalized between 0 and 1, so translate and strect
    for i in range(len(listN)):
        listN[i] = (newmax-newmin) * listN[i] + newmin
#        return listN