img_rows = 6
img_cols = 6
num_features = 13
num_images = 750
num_class = 2
x_train,y_train,x_test,y_test = read_img('......') 

x_train = x_train.reshape(num_images*img_rows*img_cols, num_features)
x_test = x_test.reshape(num_images*img_rows*img_cols, num_features)

input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 21)
y_test = keras.utils.to_categorical(y_test, 21)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test[0:10])
print(y_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, nb_epoch=100, verbose=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
