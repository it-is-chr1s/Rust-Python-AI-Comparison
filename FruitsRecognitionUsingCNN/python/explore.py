import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, os.path
import numpy as np

def sample_images():
    img = mpimg.imread('../../datasets/dataset-fruits-360/Test/Cherry Wax Black 1/164_100.jpg')
    print(img.shape)
    plt.imshow(img)
    plt.title('Test Image')
    plt.show()

    img = mpimg.imread('../../datasets/dataset-fruits-360/Training/Cherry Wax Black 1/163_100.jpg')
    print(img.shape)
    plt.imshow(img)
    plt.title('Training Image')
    plt.show()

def data_count_per_set():
    _, train_samples = getSamples('Training')
    _, test_samples = getSamples('Test')

    print("Count of Fruits in Training set:", sum(train_samples))
    print("Count of Fruits in Test set:", sum(test_samples))
    
    samples_count = sum(train_samples) + sum(test_samples)
    print("Training set: {:.2f}%".format(sum(train_samples)/samples_count*100))
    print("Test set: {:.2f}%".format(sum(test_samples)/samples_count*100))

def getSamples(set):
    categories = []
    samples = []
    for i in os.listdir('../../datasets/dataset-fruits-360/' + set + '/'):
        categories.append(i)
        samples.append(len(os.listdir('../../datasets/dataset-fruits-360/' + set + '/' + i)))
    return (categories, samples)


def distribution_of_fruits():
    train_categories, train_samples = getSamples('Training')
    test_categories, test_samples = getSamples('Test')

    figure_size = [40, 20]
    plt.rcParams["figure.figsize"] = figure_size
    index = np.arange(len(train_categories))
    plt.bar(index, train_samples)
    plt.xlabel('Fruits', fontsize=20)
    plt.ylabel('Count of Fruits', fontsize=25)
    plt.xticks(index, train_categories, fontsize=15, rotation=90)
    plt.title('Distribution of Fruits with counts in Training Set', fontsize=35)
    plt.show()

    index2 = np.arange(len(test_categories))
    plt.bar(index2, test_samples)
    plt.xlabel('Fruits', fontsize=25)
    plt.ylabel('Count of Fruits', fontsize=25)
    plt.xticks(index2, test_categories, fontsize=15, rotation=90)
    plt.title('Distrubution of Fruits with counts in Test Set', fontsize=35)
    plt.show()

def show_differences():
    train_categories, _ = getSamples('Training')
    test_categories, _ = getSamples('Test')

    train_categories = sorted(train_categories)
    test_categories = sorted(test_categories)

    print("Train categories:", len(train_categories))
    print("Test categories:", len(test_categories))

    print("Categories only in train:", set(train_categories) - set(test_categories))
    print("Categories only in test:", set(test_categories) - set(train_categories))

def main():
    #sample_images()
    data_count_per_set()
    #distribution_of_fruits()
    #show_differences()

if __name__ == "__main__":
    main()