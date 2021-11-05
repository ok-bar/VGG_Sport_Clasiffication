def observing_the_data(train,validation):
    labels=dict()
    for label_name,label_num in validation.class_indices.items():
        labels[label_num]=label_name
    plt.figure(figsize=(10,10))
    for i in tqdm(range(9)):
        plt.subplot(3,3,i+1)
        for x_batch,y_batch in validation:
            image=x_batch[0]
            argmax=np.argmax(y_batch)
            plt.tight_layout(h_pad=5)
            plt.title(labels[argmax])
            plt.xticks(())
            plt.yticks(())
            plt.imshow(image)
            break
    return plt.show()

def labels_distribution_bar(train):
    values_counter=Counter(train.classes)
    sorted(values_counter.items())
    plt.bar(train.class_indices.keys(), values_counter.values(), color=(1, 0.1, 0.1, 0.6))
    plt.xticks(rotation=90)
    return plt.show()
