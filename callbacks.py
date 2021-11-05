def callbacks():
    modelcheck=ModelCheckpoint('vgg16.h5',monitor='val_accuracy',save_best_only=True,period=1)
    earlystop=EarlyStopping(monitor='val_loss',patience=10)
    return modelcheck,earlystop
