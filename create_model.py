def model(new_model=VGG16(),layers_num=11,trainable=False):
    for layer in new_model.layers[:layers_num]:
        layer.trainable=trainable
    for i, layer in enumerate(new_model.layers):
      print(i, layer.name, layer.trainable)
    return new_model
  
def configure_model(model_name):
  flatten=model_name.layers[-4]
  predictions = model_name.layers[-1]
  dropout1 = Dropout(0.3,name='Dropout1')
  dropout2 = Dropout(0.5,name='Dropout2')
  x=Dense(units=4096,activation='relu',name='FC1',kernel_regularizer='l2')(flatten.output)
  x = dropout1(x)
  x=Dense(units=4096,activation='relu',name='FC2',kernel_regularizer='l2')(x)
  x = dropout2(x)
  predictors = Dense(22,activation='softmax',name='Predictions')(x)
  final_model = Model(inputs=model_name.input, outputs=predictors)
  print(final_model.summary())
  return final_model

