import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')


train_dataset["Fare"]=train_dataset['Fare'].fillna(train_dataset['Fare'].median())
test_dataset["Fare"]=test_dataset['Fare'].fillna(train_dataset['Fare'].median())

train_dataset["Val_Fare"] = pd.qcut(train_dataset["Fare"],4)
print (train_dataset[['Val_Fare', 'Survived']].groupby(['Val_Fare'], as_index=False).mean())

train_dataset.loc[ train_dataset['Fare'] <= 7.91, 'Fare'] = 0
train_dataset.loc[ train_dataset['Fare'] <= 14.454, 'Fare'] = 1
train_dataset.loc[ train_dataset['Fare'] <= 31, 'Fare'] = 2
train_dataset.loc[ train_dataset['Fare'] > 31, 'Fare'] = 3

test_dataset.loc[ test_dataset['Fare'] <= 7.91, 'Fare'] = 0
test_dataset.loc[ test_dataset['Fare'] <= 14.454, 'Fare'] = 1
test_dataset.loc[ test_dataset['Fare'] <= 31, 'Fare'] = 2
test_dataset.loc[ test_dataset['Fare'] > 31, 'Fare'] = 3

y = train_dataset["Survived"]
features = ["Pclass","Sex","SibSp","Parch","Fare"]
x= pd.get_dummies(train_dataset[features])
x_test = pd.get_dummies(test_dataset[features])

clf =RandomForestClassifier(n_estimators=30)
clf.fit(x,y)

predictions = clf.predict(x_test)
output = pd.DataFrame({'PassengerId':test_dataset.PassengerId, 'Survived':predictions})
output.to_csv("submission.csv",index=False)
print("\n'submission.csv' has been created\n")
