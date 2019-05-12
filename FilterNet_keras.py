from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Concatenate, AveragePooling1D, Conv1D, Flatten, BatchNormalization

wave_input = Input(shape=(24, 1))

def Conv1D_ks(kernelsize=1, in_shape=35):
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=kernelsize, input_shape=(in_shape, 1), data_format='channels_last'))
    model.add(AveragePooling1D(pool_size=2, padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    return model
    
### WaveNEet - LAYER 1:

l1_k1 = Conv1D_ks(kernelsize=1)(wave_input)
l1_k2 = Conv1D_ks(kernelsize=2)(wave_input)
l1_k3 = Conv1D_ks(kernelsize=3)(wave_input)
l1_k4 = Conv1D_ks(kernelsize=4)(wave_input)
l1_k5 = Conv1D_ks(kernelsize=5)(wave_input)
l1_k6 = Conv1D_ks(kernelsize=6)(wave_input)

### WaveNEet - LAYER 2:

l2_k1 = Conv1D_ks(kernelsize=1, in_shape=l1_k1.shape[1])(l1_k1)
l2_k2 = Conv1D_ks(kernelsize=2, in_shape=l1_k2.shape[1])(l1_k2)
l2_k3 = Conv1D_ks(kernelsize=3, in_shape=l1_k3.shape[1])(l1_k3)
l2_k4 = Conv1D_ks(kernelsize=4, in_shape=l1_k4.shape[1])(l1_k4)
l2_k5 = Conv1D_ks(kernelsize=5, in_shape=l1_k5.shape[1])(l1_k5)
l2_k6 = Conv1D_ks(kernelsize=6, in_shape=l1_k6.shape[1])(l1_k6)

dense_input = Input(shape=(31, ))

### Dense feature engineering:

x = Dense(2**10, activation="relu")(dense_input)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)

x = Dense(2**9, activation="relu")(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)

combined = Concatenate()([Flatten()(l1_k1),
                          Flatten()(l1_k2),
                          Flatten()(l1_k3),
                          Flatten()(l1_k4),
                          Flatten()(l1_k5),
                          Flatten()(l1_k6),
                          Flatten()(l2_k1),
                          Flatten()(l2_k2),
                          Flatten()(l2_k3),
                          Flatten()(l2_k4),
                          Flatten()(l2_k5),
                          Flatten()(l2_k6), x])
                          
out = Dense(1, activation="linear")(combined)

FilterNet = Model(inputs=[wave_input, dense_input], outputs=out)
