
'''
WebKB is a dataset that includes web pages from computer science departments of 
various universities. 3122 web pages are categorized into 4 imbalanced categories
(Student, Faculty, Course, Project).
'''

#importing useful libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score,roc_curve,auc,roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import plot_tree

# 1. Loading Data

# Concatinating the test and train data.Using separator as "\t" so as to get two columns viz the labels and the data 
data1=pd.read_csv(r"webkb-train-stemmed.txt",header=None,sep='\t')
data2=pd.read_csv(r"webkb-test-stemmed.txt",header=None,sep='\t')
data=pd.concat([data1,data2])

#Methord 2  Manually

#Without using pandas library 
'''
This code reads two text files, splits each line into a label and corresponding 
text, and stores all the data in a list. It’s used for loading and preprocessing 
a text dataset.
'''
dataM = []
def parse_line(line):
    parts = line.strip().split('\t')
    if len(parts) == 2:
        return tuple(parts)
    else:
        return None, None

with open('webkb-train-stemmed.txt', 'r') as file:
    for line in file:
        label, text = parse_line(line)
        if label is not None and text is not None:
            dataM.append((label, text))

with open('webkb-test-stemmed.txt', 'r') as file:
    for line in file:
        label, text = parse_line(line)
        if label is not None and text is not None:
            dataM.append((label, text))
            
  
#%%
# Open and read the text file
with open(r"webkb-train-stemmed.txt", "r") as file:
    lines_train = file.readlines()

with open(r"webkb-test-stemmed.txt", "r") as file:
    lines_test = file.readlines()

# Split lines into data and labels
data_train = []
labels_train = []
for line in lines_train:
    parts = line.strip().split("\t")
    if len(parts) == 2:  # Ensure both label and data are present
        labels_train.append(parts[0])
        data_train.append(parts[1])

data_test = []
labels_test = []
for line in lines_test:
    parts = line.strip().split("\t")
    if len(parts) == 2:  # Ensure both label and data are present
        labels_test.append(parts[0])
        data_test.append(parts[1])

# Concatenate train and test data
dataE = data_train + data_test
labelsE = labels_train + labels_test



#%%
# Exploring the data
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the last few rows of the dataset
print("\nLast few rows of the dataset:")
print(data.tail())

# Display columns of the dataset
print("\nColumns of the dataset:")
print(data.columns)

# Display shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

#------------------------------------------------------------------------------
# 3. Finding missing/null values
print(data.isna().sum())

#------------------------------------------------------------------------------

# 4. Treating missing/null values
'''
21 missing values in 3143 rows which is less than 0.67% missing data so we decide to drop the NA values 
as it is very difficult to find a specific strategy to fill corpus data.
'''

# Dropping NA values as it is very difficult to find a specific strategy to fill corpus data.
data.dropna(inplace=True)

# Checking if NA values are dropped properly
print(data.isna().sum())


#Methord 2 
# Manually drop NA values
data_no_na = []
labels_no_na = []
for i in range(len(data)):
    if dataE[i] != "na" and labelsE[i] != "na":
        data_no_na.append(dataE[i])
        labels_no_na.append(labelsE[i])

        
        
#------------------------------------------------------------------------------
# 2. Visualization of the Dataset (It will be different for text and csv files)

#Wordcloud
data[0].unique()  # Checking for labels of the data

(strstudent, strcourse, strfaculty, strproject) = ("", "", "", "")
strstudent, strcourse, strfaculty, strproject = "", "", "", ""
for i in range(len(data)):
    if data.iat[i, 0] == "student":
        strstudent = strstudent + data.iat[i, 1]

for i in range(len(data)):
    if data.iat[i, 0] == "course":
        strcourse = strcourse + data.iat[i, 1]

for i in range(len(data)):
    if data.iat[i, 0] == "faculty":
        strfaculty = strfaculty + data.iat[i, 1]

for i in range(len(data)):
    if data.iat[i, 0] == "project":
        strproject = strproject + data.iat[i, 1]

wordcloud1 = WordCloud().generate(strstudent)
wordcloud2 = WordCloud().generate(strcourse)
wordcloud3 = WordCloud().generate(strfaculty)
wordcloud4 = WordCloud().generate(strproject)

plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud of class Student")
plt.show()

plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud of class Course")
plt.show()

plt.imshow(wordcloud3, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud of class Faculty")
plt.show()

plt.imshow(wordcloud4, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud of class Project")
plt.show()


'''
Student: The word cloud suggests that web pages in the “Student” class frequently 
mention terms like “comput” (likely short for “computer” or “computing”), “science”, 
and “research”. This could indicate that these pages are related to students’ academic 
work or research in computer science.

Faculty: The word cloud for the “Faculty” class shows that terms like “computer science,” 
“research,” and “system” are common. This suggests that these pages might be about faculty 
members’ research interests or areas of expertise in computer science.

Course: In the “Course” class, words like “program,” “class,” and “project” are prominent.
 This indicates that these pages likely contain information about computer science courses, 
 such as programming assignments or class projects.
 
Project: For the “Project” class, the word cloud shows that “system,” “research,” and “project” 
are frequently used terms. This suggests that these pages might be about computer science projects, 
possibly involving systems design or research.

'''

# Plotting bar plot for class distribution
class_counts = data[0].value_counts()
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

'''
 The ‘student’ category has the highest count with over 1100 web pages. 
 The ‘faculty’ category follows with around 800 web pages.
 The ‘course’ category has approximately 600 web pages. 
 The ‘project’ category has the least number of web pages, less than 200.

'''







#%%------------------------------------------------------------------------------
# 7. Separating Data and Labels
# Separate data and labels
X = data[1]
y = data[0]



#%%------------------------------------------------------------------------------
# 6. Converting from text to numerical form

#Label Encoder
# Label Encoder Using Library
encoder_lib = LabelEncoder()
y_encoded_lib = encoder_lib.fit_transform(y)
print("Encoded Labels using Library:", y_encoded_lib)

#Manually label encode the data 
label_encoder = {}
encoded_labels = []
for label in labels_no_na:
    if label not in label_encoder:
        label_encoder[label] = len(label_encoder)
    encoded_labels.append(label_encoder[label])


#Done in subsequent steps
#CountVectoriser
#TFIDFvectoriser


#%%------------------------------------------------------------------------------
# 8. Create the Training and Testing data (This code is also implemented in subsequent steps, it is shown twice just to separate it for clarity)
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Manually train_test_split data
shuffled_data = data.sample(frac=1, random_state=42)
train_size = int(0.8 * len(shuffled_data))
train_data = shuffled_data[:train_size]
test_data = shuffled_data[train_size:]

print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))


#%% 9. Call the classsifier
# Classifier is called in the subsequent steps while applying ML algorithms
# 10. Apply all ML Algorithms

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define vectorizers
vectorizers = {
    'CountVectorizer': CountVectorizer(),
    'TF-IDF Vectorizer': TfidfVectorizer()
}


# Transform the text data
X_transformed = {vectorizer_name: vectorizer.fit_transform(X) for vectorizer_name, vectorizer in vectorizers.items()}

# Split transformed data into training and testing sets
X_train, X_test, y_train, y_test = {}, {}, {}, {}
for vectorizer_name, X_transformed_vec in X_transformed.items():
    X_train[vectorizer_name], X_test[vectorizer_name], y_train[vectorizer_name], y_test[vectorizer_name] = \
        train_test_split(X_transformed_vec, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Support Vector Machine': SVC(),
    "KNN classifier":KNeighborsClassifier(),
    'Decision tree':DecisionTreeClassifier()
}

# Initialize lists to store results
results = {vectorizer_name: {} for vectorizer_name in vectorizers.keys()}
accuracies = {vectorizer_name: {} for vectorizer_name in vectorizers.keys()}
f1_scores = {vectorizer_name: {} for vectorizer_name in vectorizers.keys()}

# Loop through each vectorizer
for vectorizer_name, vectorizer in vectorizers.items():
    # Transform the text data
    X_transformed = vectorizer.fit_transform(X)

    # Split transformed data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Loop through each classifier
    for clf_name, classifier in classifiers.items():
        # Train the classifier
        classifier.fit(X_train, y_train)

        # Predict labels
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate classification report
        clf_report = classification_report(y_test, y_pred)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[vectorizer_name][clf_name] = {'Accuracy': accuracy, 'Classification Report': clf_report, 'Confusion Matrix': conf_matrix,'F1 Score': f1}
        accuracies[vectorizer_name][clf_name] = accuracy
        f1_scores[vectorizer_name][clf_name] = f1

# Label Encoder Using Library
encoder_lib = LabelEncoder()
y_encoded_lib = encoder_lib.fit_transform(y)
print("Encoded Labels using Library:", y_encoded_lib)

#Manually label encode the data 
label_encoder = {}
encoded_labels = []
for label in labels_no_na:
    if label not in label_encoder:
        label_encoder[label] = len(label_encoder)
    encoded_labels.append(label_encoder[label])
    

#%%------------------------------------------------------------------------------
# 11. Finding Accuracy of all ML Algorithms and compare it

# Print accuracy scores
print("Accuracy Scores:")
for vectorizer_name, acc_scores in accuracies.items():
    print(f"\nVectorizer: {vectorizer_name}")
    for clf_name, acc in acc_scores.items():
        print(f"{clf_name}: {acc}")

# Visualize confusion matrix
for vectorizer_name, clf_results in results.items():
    for clf_name, result in clf_results.items():
        conf_matrix = result['Confusion Matrix']
        plt.figure()
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {clf_name} ({vectorizer_name})")
        plt.colorbar()
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

#%%------------------------------------------------------------------------------
# 12.Visualize the comparison 
# Plot results
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
metrics = ['Accuracy', 'F1 Score']

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.set_title(f'{metric} by different models with respect to both vectorizers')
    ax.set_ylabel(metric)
    ax.set_xlabel('Models')

    for vectorizer_name, results_dict in results.items():
        ax.bar(np.arange(len(results_dict)) + 0.15 * list(vectorizers.keys()).index(vectorizer_name),
               [results_dict[model][metric] for model in classifiers.keys()],
               width=0.15, label=vectorizer_name)

    ax.set_xticks(np.arange(len(classifiers)) + 0.3)
    ax.set_xticklabels(classifiers.keys(), rotation=45, ha='right')
    ax.legend()

plt.tight_layout()
plt.show()

# Print classification reports
print("\nClassification Reports:")
for vectorizer_name, clf_results in results.items():
    for clf_name, result in clf_results.items():
        print(f"\nVectorizer: {vectorizer_name} - Classifier: {clf_name}")
        print(result['Classification Report'])







#%%Visualising Decision Tree
from sklearn.tree import plot_tree



plt.figure(figsize=(20, 12), dpi=300)  
plot_tree(classifier, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=classifier.classes_)
plt.savefig('decision_tree_high_resolution.png', dpi=300)  
plt.show()

































