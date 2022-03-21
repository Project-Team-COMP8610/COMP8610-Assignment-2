import random 
from mnist import MNIST

mndata = MNIST('dataset')

trainImages, trainLabels = mndata.load_training()
# or
testImages, testLabels = mndata.load_testing()


print(trainImages[0])
# index = random.randrange(0, len(images))  # choose an index ;-)
# print(mndata.display(images[index]))
