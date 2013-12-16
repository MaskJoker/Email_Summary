import Orange     # Machine Learning Algorithm Library
import sqlite3    # SQL Python Library
import readDB as r # Read SQLite Database
from operator import itemgetter # Operation on Matrix/List of List

# Get the orignal data in the list of list format:
## [86, -0.5345224838248488, 1, 0, 1, 0, 1, 1, 1]
## [88, 0.24253562503633297, 3, 0, 1, 0, 1, 1, 1]
## [125, 0.41702882811414954, 4, 0, 1, 0, 1, 1, 1]
## [300, 0.28005601680560194, 5, 0, 1, 0, 1, 1, 1]
data = r.dataset('bc3.db')

# Pre-processing of data

mat = [list(itemgetter(4,5,6,7,12,13,15,16,3)(i)) for i in data]# without f_sa_tag

# Normalization
## average = lambda xs: sum(xs)/float(len(xs))
## for x in data.domain.features:
##     print "%-15s %.2f" % (x.name, average([d[x] for d in data]))

# Randomaization
# Traning and Testing Sets
# Cross Validation


# Feature Selection

features = []
features.append(Orange.feature.Continuous('len'))#4
features.append(Orange.feature.Continuous('senti'))#5
features.append(Orange.feature.Continuous('thr_li'))#6
features.append(Orange.feature.Continuous('rel_thr_li_n'))#7
features.append(Orange.feature.Continuous('eml_n'))#12
features.append(Orange.feature.Continuous('rel_eml_n'))#13
features.append(Orange.feature.Continuous('reply_n'))#15
features.append(Orange.feature.Continuous('recip_n'))#16

# Label

classattr = Orange.feature.Discrete('extracted', values = ['0','1'])#3

# Domain constructed through Feature and Label
domain = Orange.data.Domain(features + [classattr])

# Data
data = Orange.data.Table(domain, mat)

# Training and Testing Sets
train = data
test = data

# Data Information

### Attributes, Class, Nuumber of Data Instances
print "Attributes:", " ".join("%-15s" % x.name for x in data.domain.features)
print
print "Class:", data.domain.class_var.name
print
print "Data instances", len(data)

##  Row Information
print
print "****************Row Information******************"
### Row Information #1 (All Instances)
print "First 10 Data instances:"
print
print " ".join("%-15s" % x.name for x in data.domain.features), "%-15s" % data.domain.class_var.name
for d in data[:10]:
    print " ".join(["%-15s" % str(v) for v in d])

#### Row Information #2 (Targeted Instances)
target = "1"
print
print "Data instances with %s prescriptions:" % target
print
print " ".join("%-15s" % x.name for x in data.domain.features), "%-15s" % data.domain.class_var.name
for d in data[:10]:
    if d.get_class() == target:
        print " ".join(["%-15s" % str(v) for v in d])

### Column Information 
print
print "****************Column Information***************"
average = lambda xs: sum(xs)/float(len(xs))

print "%-15s %-15s %-15s %-15s" % ("Feature", "Mean", "Min", "Max")
for x in data.domain.features:
    print "%-15s %-15s %-15s %-15s" % (x.name, str(round(average([d[x] for d in data]),2)), str(round(min([d[x] for d in data]),2)), str(round(max([d[x] for d in data]),2)))


# Model Selection and Feature Selection

## Supervised Learning Algorithms

learnerlist = []
classifier = []
titlelist = []
metriclist = []
methodlist = []

###  1. Logistic Regression
titlelist.append("1. Logistic Regression")
methodlist.append("Logistic Regression       :")
learnerlist.append(Orange.classification.logreg.LogRegLearner())

###  2. Naive Bayseian Network
titlelist.append("2. Naive Bayseian Network")
methodlist.append("Naive Bayseian Network    :")
learnerlist.append(Orange.classification.bayes.NaiveLearner())

###  2. Naive Bayseian Network
titlelist.append("3. Support Vector Machine")
methodlist.append("Support Vector Machine    :")
learnerlist.append(Orange.classification.svm.LinearSVMLearner())

numofLearner = len(learnerlist)
for i in range(numofLearner):
    print
    print "###############################################################"
    print titlelist[i]

    #### Learning
    learner = learnerlist[i]
    
    #### Classification
    classifier = learner(train)

    #### Show Results
    print
    print "First 10 Predictions"
    for inst in test[:10]:
        classilbl = classifier(inst)
        label = inst.getclass()
        print "originally %-15s predicted %-15s"%(label, classilbl)

    ### Evaluation
    print
    print "Testing instances", len(test)

    pos = "1"
    neg = "0"

    ##### Prediction, Recall, F1, Accuracy, AUC

    m_pos = sum(1 for d in test if d.get_class() == pos)
    m_neg = sum(1 for d in test if d.get_class() == neg)

    m_true_pos = sum(1 for d in test if d.get_class() == pos and classifier(d) == pos)
    m_true_neg = sum(1 for d in test if d.get_class() == neg and classifier(d) == neg)
    m_false_pos = sum(1 for d in test if d.get_class() == neg and classifier(d) == pos)
    m_false_neg = sum(1 for d in test if d.get_class() == pos and classifier(d) == neg)

    prediction = m_true_pos*1.0/(m_true_pos + m_false_pos)
    recall = m_true_pos*1.0/(m_true_pos + m_false_neg)
    F1 = 2*prediction*recall/(prediction + recall)

    res = Orange.evaluation.testing.cross_validation([learner], data, folds=5)
    accuracy = Orange.evaluation.scoring.CA(res)[0]
    auc = Orange.evaluation.scoring.AUC(res)[0]

    print
    print "Positive: %s" % m_pos
    print "Negative: %s" % m_neg
    print
    print "                 ","%-20s"%"Confusion Matrix"
    print "                 ","%-20s %s"%("Pos", "Neg")
    print "Predicated Pos:  ","%-20s %s"%(str(m_true_pos), str(m_false_pos))
    print "Predicated Neg:  ","%-20s %s"%(str(m_false_neg), str(m_true_neg))
    print

    
    print "Prediction: %s" % prediction
    print "Recall: %s" % recall
    print "F1: %s" % F1
    print
    print "Accuracy: %.2f" % accuracy
    print "AUC:      %.2f" % auc
    print

    metriclist.append([auc, F1, accuracy, prediction, recall] )
    
## Evaluation Presentation
print

### Metrics Data
metricdata = metriclist

### Metrics Name
metricname = []
metricname.append('AUC')
metricname.append('F1')
metricname.append('Accuracy')
metricname.append('Prediction')
metricname.append('Recall')


### Learning Method Comparison
print "------------------------------Learning Methods Comparison----------------------------"
print
print "                           "," ".join("%-15s" % x for x in metricname)
for d in range(len(metricdata)):
    print "%-25s" % methodlist[d]," ".join(["%-15s" % str(round(v,2)) for v in metricdata[d]])

