print("Hello, GitHub")
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import scipy
print(scipy.__version__)
t=np.array(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
p=pd.Series(t)
print(p)
import matplotlib.pyplot as plt
x = [1, 5, 7]
y = [20, 40, 60]
plt.plot(x, y)
plt.xlabel('x-axis')
text = (0.5, 0, 'xaxis')
plt.ylabel('y-axis')
text = (0, 0.5, 'yaxis')
plt.title('Value of y coresponding to x')
text = (0.5, 1.0, 'Value of y coresponding to x')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)

import os
#Look inside the download folder to find your csv file
files = os.listdir(path)
print("Files inside the folder:", files)
csv_path = os.path.join(path, "spam.csv")
df = pd.read_csv(csv_path, encoding='latin-1')
#View the data
print(df.head())

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Remove leading and trailing whitespaces from column names (Optional)
df.columns = df.columns.str.strip()
# Drop the specified columns (Optional)
columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
df.drop(columns=columns_to_drop, inplace=True)

# Rename columns
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)

# Add a binary column indicating whether the message is spam or not
df["spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["spam"], test_size=0.25, random_state=42)

# Convert text data into numerical vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Test the model on sample Messages 
messages = [
    "Nah I don't think he goes to us, he lives around here though",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]
messages_count = vectorizer.transform(messages)
predictions = model.predict(messages_count)  # 1 for spam, 0 for not spam

# Print predictions
for i, message in enumerate(messages):
    if predictions[i] == 0:
        print(f"'{message}' ---> Normal Message")
    else:
        print(f"'{message}' ---> Spam Message")

# Evaluate model accuracy on the test set
accuracy = model.score(X_test_count, y_test)
print("Model Accuracy:", accuracy)
