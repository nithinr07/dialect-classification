import warnings
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import csv
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb


# ## Loading an audio

# In[80]:


warnings.filterwarnings("ignore")
audio_path = 'dataset/belfast/1.wav'
x, sr = librosa.load(audio_path)
print(sr)


# In[81]:


ipd.Audio(audio_path)


# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[84]:


#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# ## Feature Extraction

# In[129]:


#MFCC
mfccs = librosa.feature.mfcc(x, sr=sr)
mfccs_delta = librosa.feature.delta(mfccs)
mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
print(mfccs.shape)
print(mfccs_delta.shape)
print(mfccs_delta2.shape)
#Displaying  the MFCCs:
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(mfccs)
plt.title('MFCC')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(mfccs_delta)
plt.title(r'MFCC-$\Delta$')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(mfccs_delta2, x_axis='time')
plt.title(r'MFCC-$\Delta^2$')
plt.colorbar()
plt.tight_layout()
plt.show()


# In[123]:


#Spectral Flux
spectral_flux = librosa.onset.onset_strength(y=x, sr=sr)
print(spectral_flux)
# D = np.abs(librosa.stft(x))
# times = librosa.times_like(D)
# plt.plot(times, 2 + spectral_flux / spectral_flux.max(), alpha=0.8, label='Mean (mel)')


# ## Building the dataset

# In[130]:


header = ''
for i in range(1, 21):
    header += f' mfcc_{i}'
for i in range(1, 21):
    header += f' mfcc_delta_{i}'
for i in range(1, 21):
    header += f' mfcc_delta2_{i}'
header += ' spectral_flux'
header += ' label'
header = header.split()
print(header)


# In[131]:


file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
dialects = 'belfast bradford cambridge dublin leeds london newcastle cardiff liverpool'.split()
for g in dialects:
    for filename in os.listdir(f'./dataset/{g}'):
        audio = f'./dataset/{g}/{filename}'
        y, sr = librosa.load(audio)
        to_append = f''
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        for e in mfccs:
            to_append += f' {np.mean(e)}'
        mfccs_delta = librosa.feature.delta(mfccs)
        for e in mfccs_delta:
            to_append += f' {np.mean(e)}'
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        for e in mfccs_delta2:
            to_append += f' {np.mean(e)}'
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        to_append += f' {np.mean(spectral_flux)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
    print(g)


# ## Creating the dataframe

# In[141]:


data = pd.read_csv('data.csv')
data.head()


# In[142]:


dialect_list = data.iloc[:, -1]
encoder = LabelEncoder()
labels = encoder.fit_transform(dialect_list)
print(labels)


# In[143]:


data.describe()


# In[252]:


data.isnull().sum()


# In[257]:


data1 = data.iloc[:,:20]
data1['spectral_flux'] = data['spectral_flux']
data1.head()


# In[264]:


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:,:-1], dtype = float))
# X = scaler.fit_transform(np.array(data.iloc[:,:20], dtype = float))
# X = scaler.fit_transform(np.array(data1, dtype = float))


# ## Model Building

# In[265]:


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)


# In[271]:


#Logistic Regression
logistic_classifier = linear_model.logistic.LogisticRegression()
logistic_classifier.fit(X_train, y_train)
logistic_predictions = logistic_classifier.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_cm = confusion_matrix(y_test, logistic_predictions)
print("logistic accuracy = " + str(logistic_accuracy))
print(logistic_cm)
print(classification_report(y_test, logistic_predictions))


# In[272]:


#SVM
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_cm = confusion_matrix(y_test, svm_predictions)
print("svm accuracy = " + str(svm_accuracy))
print(svm_cm)
print(classification_report(y_test, svm_predictions))


# In[273]:


#KNN
knn_classifier = KNeighborsClassifier(n_neighbors = 3)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
print("knn accuracy = " + str(knn_accuracy))
print(knn_cm)
print(classification_report(y_test, knn_predictions))


# In[278]:


rf_classifier = RandomForestClassifier(n_estimators = 5000, random_state = 42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
print("rf accuracy = " + str(rf_accuracy))
print(rf_cm)
print(classification_report(y_test, rf_predictions))


# In[274]:


#XGBoost
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)
nc = labels.size
param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': nc} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])
xgboost_accuracy = accuracy_score(y_test, best_preds)
xgboost_cm = confusion_matrix(y_test, best_preds)
print("XGBoost accuracy = " + str(xgboost_accuracy))
print(xgboost_cm)
print(classification_report(y_test, best_preds))

