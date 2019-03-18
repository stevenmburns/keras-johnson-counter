import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

#
# How about all 2**n
#
n = 16
ss = []
for k in range(1<<n):
    ss.append( [ 1 if k & (1<<i) else 0 for i in range(n)])

def ns( v):
    return [1-v[-1]] + v[:-1]

x_train = np.array( ss, dtype=np.float64)
y_train = np.array( [ ns(v) for v in ss], dtype=np.float64)


a = tf.keras.layers.Input( shape=(n,))

split = [ tf.keras.layers.Lambda( lambda x: x[:, i:(i+1)])(a) for i in range(n)]

bs = [ tf.keras.layers.Dense( 1)(split[(i-1)%n]) for i in range(n)]

merge = tf.keras.layers.Concatenate()( bs)

c = tf.keras.layers.Activation( 'sigmoid')(merge)

model = tf.keras.models.Model( inputs=a, outputs=c)

indices = list(range(len(x_train)))

#random.shuffle(indices)

n_val = len(indices)//5

x_val,x_train = x_train[indices[:n_val]],x_train[indices[n_val:]]
y_val,y_train = y_train[indices[:n_val]],y_train[indices[n_val:]]

len(x_train)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.2),
              loss='binary_crossentropy',
              metrics=['accuracy', 'mse', 'mae']
              )


model.fit(x_train,y_train,epochs=8,validation_data=(x_val,y_val))

def pv(v):
    return ' '.join( "%.2f" % x for x in v)

x_test = x_val
for k,v in zip(x_test,model.predict(x_test)):
    print( pv(k), "=>", pv(v))

plt.plot( model.history.history['mae'], "g")
plt.plot( model.history.history['val_mae'], "or")

plt.show()
