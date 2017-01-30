import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  prior_probability = util.Counter() # to store the prior probability

  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    # count_feature and total_feacture have the form of [label][valeu][feature]
    count_feature = {} # keep counting the features swith different value of (0,1)
    total_feature = {} # total counting of features
    best_result_conditional = {} # to store the conditional probability that give the best result due to k value
    highestPrediction = None # to store the accuracy rate for the prediction
    correct_prediction = 0.0 # to store number of prediciton that is correct
    # prepare the store dataset to calculate conditional probability and also prior probability
    for label in trainingLabels:
      self.prior_probability[label] += 1.0
      count_feature[label] = util.Counter()
      total_feature[label] = util.Counter()
      for value in [0,1]:
        count_feature[label][value]= util.Counter()
    #Calculate prior probability
    self.prior_probability.normalize()
    # start counting all the possible feature with the possible value of (0,1)
    for i in range(len(trainingData)):
      label = trainingLabels[i]
      dataSet = trainingData[i]
      for feature, value in dataSet.items():
        count_feature[label][value][feature]+=1.0
        total_feature[label][feature]+=1.0
    # start calculating the conditional probability with Laplace smoothing algorithm
    # conditional probability also have the same form of prior probability which is
    # conditional[label][value][feature]
    for k in kgrid or [0.0]:
      conditional = {}# to store all the temporary conditional probability with temporary k value
      correct_prediction = 0.0 # number of correct guess
      for label in self.legalLabels:
        conditional[label] = util.Counter()
        for value in [0,1]:
          conditional[label][value] = util.Counter() 
          for feature in self.features:
            conditional[label][value][feature] = (count_feature[label][value][feature] + k*1.0)/(total_feature[label][feature]+2.0*k)
      self.conditional = conditional
      prediction = self.classify(validationData) # store the prediction from validation data to compare with validationLabel
      #for index in range(len(prediction)):
      for i in range(len(prediction)):
      	if validationLabels[i] == prediction[i] :
      		correct_prediction += 1.0
        #if validationLabels[i] == prediction[index]: # compare validation and prediction
         # correct_prediction += 1.0
      accuracy = correct_prediction / len(prediction) # calculate the accuracy rate
      if (accuracy > highestPrediction) or (highestPrediction == None):
        highestPrediction = accuracy # replace the highest prediction rate
        best_result_conditional = conditional.copy() # replace the best set of conditional probability
        self.k = k 
        
    self.conditional = best_result_conditional.copy() # final form of the conditional probability
    
    
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    
    logJoint = util.Counter()
    evidence = datum.items()
    "*** YOUR CODE HERE ***"
    for label in self.legalLabels:
      logJoint[label] = math.log(self.prior_probability[label]) # inital with prior probability
      for feature in self.features:
          value = datum[feature]
          conLogProba = self.conditional[label][value][feature]
          if conLogProba == 0: # to avoid the case of conLogProba is 0 which is lead to earror for log calculation
            logJoint[label] += 0.0 # assign float number of 0.0 in case of conditional is 0
          else:
            logJoint[label] += math.log(conLogProba) # plus the log of conditional probability to logJoint
    return logJoint