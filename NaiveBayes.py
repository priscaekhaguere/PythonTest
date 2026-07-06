import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# read the labeled dataset
train_data = pd.read_csv('train-data.csv')

print('Shape of full dataset :', train_data.shape)

# drop columns that aren't useful for prediction (text-heavy, high missing values, or IDs)
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# fill missing Age values with the median age
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# fill missing Embarked values with the most common value
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# convert 'Sex' to numeric: male=0, female=1
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# convert 'Embarked' to numeric using one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# separate features and target
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# split into train/test ourselves
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(train_x, train_y)

predict_train = model.predict(train_x)
print('Target on train data', predict_train)
accuracy_train = accuracy_score(train_y, predict_train)
print('accuracy_score on train dataset :', accuracy_train)

predict_test = model.predict(test_x)
print('Target on test data', predict_test)
accuracy_test = accuracy_score(test_y, predict_test)
print('accuracy_score on test dataset :', accuracy_test)