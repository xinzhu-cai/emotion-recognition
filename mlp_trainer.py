from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import keras.callbacks
import numpy as np
import data_reader
import random

class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.accs.append(logs.get('acc'))


np.random.seed(1234567)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
model.add(Dense(64, activation='relu', input_dim=data_reader.fft_n*4))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
history = AccHistory()

result = open("result.txt", 'w')

all_seq, all_label, all_name = data_reader.read_all()
y = all_label[0]
x = all_seq[0]
name = all_name[0]

acc_history=[]
x_train = []
y_train = []
x_test = []
y_test = []
test_name = []
selects = []

ks = 5.0

for j in range(ks):
    num = random.randint(0, 35)
    while num in selects:
        num = random.randint(0, 35)
    selects.append(num)
for j in range(36):
    if j in selects:
        x_test.append(x[j])
        y_test.append(y[j])
        test_name.append(name[j])
    else:
        x_train.append(x[j])
        y_train.append(y[j])
for i in range(12): #10 rounds

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print x_train.shape
    model.fit(x_train, y_train, epochs=100, batch_size=16,shuffle=True, verbose=0, callbacks=[history])
    if i>1:
        for k in range(ks):
            score = model.evaluate(x_test[k:k+1], y_test[k:k+1], batch_size=5, verbose=0)
            acc_history.append(score[1])
            if score[1]<0.1:
                print test_name[k], "true_result:", y_test[k]
    score = model.evaluate(x_test, y_test, batch_size=5, verbose=0)
    print "test "+str(i+1)+str(score)

sums = 0
for i in range(36):
    score = model.evaluate(x[i:i + 1], y[i:i + 1], batch_size=1, verbose=0)
    result.write(name[i] + '\t' + str(y[i]) + '\t' + str(score[1]) + '\n')
    sums += int(score[1])
print sums/36.0

score = model.evaluate(x, y, batch_size=36)
print "final test " + str(score)
print sum(acc_history)/(ks*10)