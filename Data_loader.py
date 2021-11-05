file_path = '....'

def data_pre_processing(file_path,valid_split = 0.25,input_size = (224, 224),image_color = 'rgb',batch_size = 32,
                        shuffle=True ):
 
    train_gen=ImageDataGenerator(rescale=1./255,validation_split=valid_split,zoom_range=0.5,horizontal_flip=True,rotation_range=40,vertical_flip=0.5,width_shift_range=0.3,height_shift_range=0.2,brightness_range=[0.2,1.0],fill_mode='nearest')

    validation_gen=ImageDataGenerator(rescale=1./255,validation_split=valid_split)

    train_data=train_gen.flow_from_directory(directory=file_path,target_size=input_size,color_mode=image_color,
                                             batch_size=batch_size,shuffle=shuffle,subset='training')
    test_data=validation_gen.flow_from_directory(directory=file_path,target_size=input_size,color_mode=image_color,
                                             batch_size=batch_size,shuffle=shuffle,subset='validation')

    return train_data,test_data
