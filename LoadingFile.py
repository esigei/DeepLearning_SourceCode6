from __future__ import print_function

from keras.engine.saving import load_model

from DeepLearning_SourceCode6.inclass import Y_train, X_train

model=load_model('model.h5')
print(model.evaluate(X_train,Y_train))