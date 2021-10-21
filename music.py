import tensorflow as tf
import numpy as np
import pandas as pd  # to work with data
import matplotlib.pyplot as plt #visualization
import seaborn as sns #
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


path_train = 'PATH_TO_TRAIN_CSV'
path_test = 'PATH_TO_TEST_CSV'

df = pd.read_csv(path_train)
df = df.drop(columns=['type'])
print(df.describe(include='all'))
print(df.isnull().sum())
print(df.info())

def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = round((df.isnull().sum())/(df.isnull().count()),3)*100
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

print(check_missing_data(df))

df.plot.hist('Rhythm')
df.hist('vibrance')
df.hist('key')
df.hist('Decibel_Levels')
df.hist('mode')
df.hist('lyrics_amount')
df.hist('acoustics')
df.hist('instruments')
df.hist('bounce')
df.hist('valence')
df.hist('Beats_Speed')
df.hist('TimeLength')
df.hist('Hyperactivity')
df.hist('MusicEraRating')
# plt.show()

df['Rhythm'] = df['Rhythm'].fillna(df.Rhythm.median())
df['vibrance'] = df['vibrance'].fillna(df.vibrance.median())
df['key'] = df['key'].fillna(df.key.mean())
df['Decibel_Levels'] = df['Decibel_Levels'].fillna(df.Decibel_Levels.median())
df['lyrics_amount'] = df['lyrics_amount'].fillna(df.lyrics_amount.median())
df['acoustics'] = df['acoustics'].fillna(df.acoustics.median())
df['instruments'] = df['instruments'].fillna(df.instruments.mean())
df['bounce'] = df['bounce'].fillna(df.bounce.median())
df['valence'] = df['valence'].fillna(df.valence.median())
df['Beats_Speed'] = df['Beats_Speed'].fillna(df.Beats_Speed.median())
df['TimeLength'] = df['TimeLength'].fillna(df.TimeLength.median())
df['Hyperactivity'] = df['Hyperactivity'].fillna(df.Hyperactivity.median())
df['MusicEraRating'] = df['MusicEraRating'].fillna(df.MusicEraRating.mean())

print(check_missing_data(df))

df['title'] = df['title'].fillna(df.title.mode()[0])
df['mode'] = df['mode'].fillna(df['mode'].mode()[0])

print(check_missing_data(df))

plt.figure(figsize=(12, 10)) # Set the figure size
sns.heatmap(df.corr(), annot=True) # Print the heatmap
# plt.show()
#hyper-valence , instru-timeleng , deci - vib

print(df[['title','genre']].nunique())

for i in ['title','genre']:
  print(f'{i} - {df[i].unique()}', end='\n\n')

from sklearn.preprocessing import LabelEncoder
data = df.copy()
labelenc = LabelEncoder()
data['genre'] = labelenc.fit_transform(data['genre'])
print(data.iloc[:10,:])

def makeCountPlot(data, width=0, height=4, hue=None):
  if width == 0:
    width = len(data.unique())*1.25
  plt.figure(figsize=(width, height))
  sns.countplot(x=data, hue=hue) # to see how y value varies
#   plt.show()

makeCountPlot(data=df['genre'])

df = pd.get_dummies(df,columns=['title'])
print(df)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
#stdscaler - To reduce outliers and make the distributuion uniform. It helps increase the accuracy fo the model
from sklearn.model_selection import train_test_split

x = df.drop(columns=['genre'])
y = np.array(data['genre'], dtype='float32')
print(y)

mms = MinMaxScaler() # Creating an instance of this class
ss = StandardScaler()
x = ss.fit_transform(x)
x = mms.fit_transform(x)
x = pd.DataFrame(x, columns=df.drop(columns=['genre']).columns)
print(x)

x_tr, x_val, y_tr, y_val = train_test_split(x, y,train_size=0.7 ) #stratify is used to get the ratio of splitting the data

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# model = GaussianNB()
model = DecisionTreeClassifier()
model.fit(x_tr, y_tr)

pred = model.predict(x_val)
print(classification_report(pred,y_val))

nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(x_tr.shape[1], input_shape=(x_tr.shape[1], ), activation='relu'),
    tf.keras.layers.Dense(x_tr.shape[1]*2),
    tf.keras.layers.Dense(x_tr.shape[1]*3),
    tf.keras.layers.Dense(x_tr.shape[1]*2, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])
nn.compile(optimizer=tf.keras.optimizers.Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
nn.summary()

nn.fit(x=x_tr,y=y_tr,validation_data=(x_val,y_val),epochs=40)

nnpred = nn.predict(x_val)
ls = list(nnpred[0])
max_value = max(ls)
max_index = ls.index(max_value)
classes = ['Dark Trap','Rhythm & Blues','Hiphop','hardstyle','trance','Drums & Bass','Hipstter-Hop','Electro','psytrance','Pop','trap','Emotional','Rap','techno','Industrial Trap']
classes.sort()
print(max_value,classes[max_index])
print(nnpred)

pred = []
for i in range(len(x_val)):
  ls = list(nnpred[i])
  max_value = max(ls)
  max_index = ls.index(max_value)
  pred.append(max_index)

print(classification_report(pred, y_val))

#TEST DATA:
test_df = pd.read_csv(path_test)
print(test_df.head())

test_df = test_df.drop(columns=['type','Usage'])
print(test_df.info())

print(test_df.describe(include='all'))

for i in ['title']:
  print('{0} - {1}'.format(i,test_df[i].unique()))

id = test_df['id']
test_df = test_df.drop(columns=['id'])
print(id)

print(test_df)

print(check_missing_data(test_df))

test_df.hist('Rhythm')
test_df.hist('vibrance')
test_df.hist('key')
test_df.hist('Decibel_Levels')
test_df.hist('mode')
test_df.hist('lyrics_amount')
test_df.hist('acoustics')
test_df.hist('instruments')
test_df.hist('bounce')
test_df.hist('valence')
test_df.hist('Beats_Speed')
test_df.hist('TimeLength')
test_df.hist('Hyperactivity')
test_df.hist('MusicEraRating')
# plt.show()

test_df['Rhythm'] = test_df['Rhythm'].fillna(test_df.Rhythm.median())
test_df['vibrance'] = test_df['vibrance'].fillna(test_df.vibrance.median())
test_df['key'] = test_df['key'].fillna(test_df.key.mean())
test_df['Decibel_Levels'] = test_df['Decibel_Levels'].fillna(test_df.Decibel_Levels.median())
test_df['lyrics_amount'] = test_df['lyrics_amount'].fillna(test_df.lyrics_amount.median())
test_df['acoustics'] = test_df['acoustics'].fillna(test_df.acoustics.median())
test_df['instruments'] = test_df['instruments'].fillna(test_df.instruments.mean())
test_df['bounce'] = test_df['bounce'].fillna(test_df.bounce.median())
test_df['valence'] = test_df['valence'].fillna(test_df.valence.median())
test_df['Beats_Speed'] = test_df['Beats_Speed'].fillna(test_df.Beats_Speed.median())
test_df['TimeLength'] = test_df['TimeLength'].fillna(test_df.TimeLength.median())
test_df['Hyperactivity'] = test_df['Hyperactivity'].fillna(test_df.Hyperactivity.median())
test_df['MusicEraRating'] = test_df['MusicEraRating'].fillna(test_df.MusicEraRating.mean())  

test_df['title'] = test_df['title'].fillna(test_df.title.mode()[0])
test_df['mode'] = test_df['mode'].fillna(test_df['mode'].mode()[0])

print(check_missing_data(test_df))

print(test_df)

plt.figure(figsize=(12, 10)) # Set the figure size
sns.heatmap(test_df.corr(), annot=True) # Print the heatmap
# plt.show()

print(test_df[['title']].nunique())

test_df = pd.get_dummies(test_df,columns=['title'])
ls=[0]*len(id)
test_df['title_Techno Bangers: Bunker Style'] = ls
test_df['title_Hardstlye - 2020']=ls

print(test_df)

x_test = ss.fit_transform(test_df)
x_test = mms.fit_transform(x_test)
print(x_test)

nnpred_test = nn.predict(x_test)
ls = list(nnpred_test[0])
max_value = max(ls)
max_index = ls.index(max_value)
classes = ['Dark Trap','Rhythm & Blues','Hiphop','hardstyle','trance','Drums & Bass','Hipstter-Hop','Electro','psytrance','Pop','trap','Emotional','Rap','techno','Industrial Trap']
classes.sort()
print(max_value,classes[max_index])
print(nnpred_test)


pred_test = []
for i in range(len(x_test)):
  ls = list(nnpred_test[i])
  max_value = max(ls)
  max_index = ls.index(max_value)
  pred_test.append(classes[max_index])
print(pred_test)


index_val=[i for i in range(len(id))]
# print(test_df.shape()[0])
id = pd.Series(id)
pred_test = pd.Series(pred_test,name='genre')
data_df = pd.concat([id,pred_test],axis=1)
print(data_df)

# data_df.to_csv(path_or_buf=r'./Datahub-2021/submission.csv',columns=['id','genre'],index=False)

