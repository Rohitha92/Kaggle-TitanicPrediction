

import pandas as pd
from sklearn import svm
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualizations
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']= 8,6

##############################################################
#load data
train = pd.read_csv("train.csv")

#get details of the dataset
dataframe = pd.DataFrame(train.dtypes)
print dataframe
print train.head()

#check for null values in each column
print train.isnull().sum()
#Age =177, Cabin=687 and Embarked=2 as null values
#Need to fix the null values. First analyse the importance of
#each features before dropping them

#Find how many survived in train set. use sns to count
sns.countplot(x='Survived', data =train)
plt.show()
#only 342 survived out of 891 passengers.

'''
CATEGORICAL FEATURES : Sex, Embarked

ORDINAL FEATURES: Pclass
same as categorical but have relative ordering like short medium tall.

CONTINOUS FEATURE: Age 
'''


#Anlyzing the features

'''Sex category vs Survived'''

print train.groupby(['Sex', 'Survived'])['Survived'].count()


sns.countplot('Sex', hue= 'Survived', data=train)
plt.show()


'''Pclass and Pclass Vs Sex and Survived'''

sns.countplot('Pclass', hue='Survived', data=train)
plt.show()
print pd.crosstab([train.Sex, train.Survived], train.Pclass, margins= True)

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
#sns.factorplot('Pclass', 'Survived', hue= 'Sex', data=train, ax= axs[0])
sns.countplot('Pclass',hue='Sex', data=train, ax=axs[0])
axs[0].set_title('Male-Female Count in Pclass')
sns.barplot(x='Sex', y='Survived', hue= 'Pclass', data=train, ax= axs[1])
axs[1].set_title('Male-Female Survival Vs Pclass')
plt.show()


'''Age'''

#Initial details
print train['Age'].max()
print train['Age'].min()
print train['Age'].mean()

#Extract Title from Name
train['Title'] = 0 #create another column
for i in train:
    train['Title'] = train.Name.str.extract('([A-Za-z]+)\.')
print pd.crosstab(train.Title, train.Sex)

train['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major',
                        'Master','Miss','Mr','Mrs','Ms','Rev','Sir','Mme', 'Mlle'],
            ['Other','Mr','Mrs','Mr','Mr','Other','Mrs','Mr','Master','Miss','Mr','Mrs', 'Miss','Other','Mr',
             'Miss', 'Miss'],
                       inplace=True)

print pd.crosstab(train.Title, train.Sex)

#Get mean age of each title.
print train.groupby('Title').Age.mean()

print train.head()
#Fill in the missing values with these average means w.r.t Titles
train.loc[(train.Age.isnull()) & (train.Title =='Mr'),'Age'] = 33
train.loc[(train.Age.isnull()) & (train.Title =='Miss'),'Age'] = 22
train.loc[(train.Age.isnull()) & (train.Title =='Master'),'Age'] = 5
train.loc[(train.Age.isnull()) & (train.Title =='Mrs'),'Age'] = 36
train.loc[(train.Age.isnull()) & (train.Title =='Other'),'Age'] = 46

#check for null values
print train.Age.isnull().any()

#plot Age vs survived
g=sns.FacetGrid(col= 'Survived', row= 'Sex', data=train, margin_titles=True)
g.map(plt.hist, 'Age', color = 'green')
g.set_xticklabels(rotation=30)
plt.show()

'''Embarked'''
#Check the survival rate
fig, axs = plt.subplots(1,3, figsize=(8,18))
sns.factorplot('Embarked', 'Survived', data=train, ax=axs[0])
axs[0].set_title('Embarked Vs Survived')
sns.barplot(x='Embarked',y='Survived', hue='Pclass',data=train, ax=axs[1])
axs[1].set_title('Pclass Survived at Embark ')
sns.countplot('Embarked', hue='Pclass', data=train, ax=axs[2])
axs[2].set_title('No.of Pclass Embarked')
plt.show()

#Filling NAN values with class S
train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'
train.Embarked.isnull().any()


'''SibSp and Parch'''
fig,axs= plt.subplots(2,2, figsize=(20,22))
sns.countplot('SibSp', hue= 'Survived', data=train, ax=axs[0,0])
axs[0,0].set_title('SibSp vs Survived')
sns.countplot('Parch', hue= 'Survived', data=train, ax=axs[0,1])
axs[0,1].set_title('Parch Vs Survived')
sns.stripplot(x='SibSp', y='Age', hue= 'Survived', data=train, ax=axs[1,0])
axs[1,0].set_title('SibSp Vs Age and Survived')
sns.stripplot(x='Parch', y='Age', hue= 'Survived', data=train, ax=axs[1,1])
axs[1,1].set_title('Parch Vs Age and Survived')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()

fig,axs= plt.subplots(1,2, figsize=(8,18))
sns.barplot(x='SibSp',y='Survived', hue= 'Pclass', data= train, ax= axs[0])
axs[0].set_title('SibSp Vs Pclass and Survived')
sns.barplot(x='Parch',y='Survived', hue= 'Pclass', data= train, ax= axs[1])
axs[1].set_title('Parch Vs Pclass and Survived')
plt.show()


'''Correlation'''

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

'''COnvert Age into bins'''

train['Age_bins']=0
train.loc[train['Age']<=10,'Age_bins']=0
train.loc[(train['Age']>10)&(train['Age']<=20),'Age_bins']=1
train.loc[(train['Age']>20)&(train['Age']<=30),'Age_bins']=2
train.loc[(train['Age']>30)&(train['Age']<=40),'Age_bins']=3
train.loc[(train['Age']>40)&(train['Age']<=50),'Age_bins']=4
train.loc[(train['Age']>60)&(train['Age']<=70),'Age_bins']=5
train.loc[train['Age']>70,'Age_bins']=6
print train.head()

'''Combine Parch and SibSp'''

train['Family']=0
train['Family']=train['Parch'] + train['SibSp']
print train.head()

'''Convert Sex, Embark into numeric categorical'''

train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

'''Drop Passenger_ID,Age, Name,Fare, Cabin, Title, Parch, Ticket, SibSp'''
train.drop(['Name','Age','Ticket','Title','Fare','Cabin','PassengerId', 'SibSp','Parch'],axis=1,inplace=True)

print train.head()

'''Correlation matrix of the final Features'''
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

####################################################################33
''' Read in test.csv and convert it suitably'''
test = pd.read_csv("test.csv")
print test.head()
print test.isnull().sum()

test['Title'] = 0 #create another column
for i in test:
    test['Title'] = test.Name.str.extract('([A-Za-z]+)\.')

test['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major',
                        'Master','Miss','Mr','Mrs','Ms','Rev','Sir','Mme', 'Mlle'],
            ['Other','Mr','Mrs','Mr','Mr','Other','Mrs','Mr','Master','Miss','Mr','Mrs', 'Miss','Other','Mr',
             'Miss', 'Miss'],
                       inplace=True)

test.loc[(test.Age.isnull()) & (test.Title =='Mr'),'Age'] = 33
test.loc[(test.Age.isnull()) & (test.Title =='Miss'),'Age'] = 22
test.loc[(test.Age.isnull()) & (test.Title =='Master'),'Age'] = 5
test.loc[(test.Age.isnull()) & (test.Title =='Mrs'),'Age'] = 36
test.loc[(test.Age.isnull()) & (test.Title =='Other'),'Age'] = 46

test['Age_bins']=0
test.loc[test['Age']<=10,'Age_bins']=0
test.loc[(test['Age']>10)&(test['Age']<=20),'Age_bins']=1
test.loc[(test['Age']>20)&(test['Age']<=30),'Age_bins']=2
test.loc[(test['Age']>30)&(test['Age']<=40),'Age_bins']=3
test.loc[(test['Age']>40)&(test['Age']<=50),'Age_bins']=4
test.loc[(test['Age']>60)&(test['Age']<=70),'Age_bins']=5
test.loc[test['Age']>70,'Age_bins']=6
print test.head()

test['Family']=0
test['Family']=test['Parch'] + test['SibSp']
print test.head()

test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

'''Drop Passenger_ID,Age, Name,Fare, Cabin, Title, Parch, Ticket, SibSp'''
test.drop(['Name','Age','Ticket','Title','Fare','Cabin','PassengerId', 'SibSp','Parch'],axis=1,inplace=True)

print test.head()


#################################################################
'''Training and Testing Machine learning model for prediction'''
train,test=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=train[train.columns[1:]]
Y=train['Survived']


model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))


'''KNN '''
model=KNeighborsClassifier()
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
