from LetterDecomposer import DecomposeLetter
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from keras import models
from keras import layers

#데이터 입수


tNews = pd.read_csv("data/True.csv")
fNews = pd.read_csv("data/Fake.csv")


zeroVector = pd.read_csv("data/freq.csv")


v0 = zeroVector.loc[:, "frequency"].to_numpy()
print("v0:", v0)


#test

tnList = list()
fnList = list()

n11 = list()
n12= list()
n13 = list()

vectorList = list()


for n in range(len(fNews.iloc[:, 0])):

    if n % 100 == 0 and n != 0:

        print("%dth fake news evaluation."%n)

    text = fNews.iloc[n, :]["text"]


    text = DecomposeLetter(text)
    text.decompose()
    v = text.vector() - v0


    vectorList.append(v)


n21 = list()
n22= list()
n23 = list()
vectorList2 = list()

for n in range(len(tNews.iloc[:, 0])):

    if n % 100 == 0 and n != 0:
        print("%dth True news evaluation."%n)

    text = tNews.iloc[n, :]["text"]

    text = DecomposeLetter(text)
    text.decompose()
    v = text.vector() - v0


    vectorList2.append(v)



fData = pd.DataFrame(np.array(vectorList))
tData = pd.DataFrame(np.array(vectorList2))

fData.insert(26, "label", 0)
tData.insert(26, "label", 1)

finalData = fData.append(tData)

finalData = finalData.sample(frac=1)

trainingData = finalData.iloc[:30000, :-1]
trainingLabel = finalData.iloc[:30000, :].loc[:, "label"]

trainingData2 = finalData.iloc[30000:35000, :-1]
trainingLabel2 = finalData.iloc[30000:35000, :].loc[:, "label"]
trainingData3 = finalData.iloc[35000:, :-1]
trainingLabel3 = finalData.iloc[35000:, :].loc[:, "label"]

print(finalData)

train_x = trainingData.to_numpy()

val_x = trainingData2.to_numpy()
val_y = trainingLabel2.to_numpy()

eva_x = trainingData3.to_numpy()
eva_y = trainingLabel3.to_numpy()

train_y = trainingLabel.to_numpy()

def modelGenerate(nodes, hiddenLayers):

    model = models.Sequential()
    model.add(layers.Dense(nodes, activation='relu', input_shape=(26,)))

    for n in range(hiddenLayers):

        model.add(layers.Dense(nodes, activation='relu'))



    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


model1 = modelGenerate(256, 3)
model2 = modelGenerate(128, 2)
model3 = modelGenerate(64, 1)

model1.fit(train_x, train_y, epochs=500, batch_size=50, validation_data=(val_x, val_y))
model2.fit(train_x, train_y, epochs=500, batch_size=50, validation_data=(val_x, val_y))
model3.fit(train_x, train_y, epochs=500, batch_size=50, validation_data=(val_x, val_y))

val_loss, val_acc = model1.evaluate(eva_x, eva_y, batch_size=1)



val_loss2, val_acc2 = model1.evaluate(eva_x, eva_y, batch_size=1)



val_loss3, val_acc3 = model1.evaluate(eva_x, eva_y, batch_size=1)
print("256 Nodes, 3 hidden Layers", val_loss, val_acc)
print("128 Nodes, 2 hidden Layers", val_loss2, val_acc2)
print("64 Nodes, 1 hidden Layers", val_loss3, val_acc3)
