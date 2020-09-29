import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SOM:

    def __init__(self, height, width, input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])
        
        #get location width closest distance
        best_matching_unit = self.get_bmu()

        #update weight
        self.updated_weight, self.rate_stacked = self.update_neighbour(best_matching_unit)

    def train(self, dataset, num_of_epoch):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range (num_of_epoch):
                for data in dataset:
                    sess.run(self.updated_weight, feed_dict={ self.input: data})
            
            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            for i, loc in enumerate(location):
                print(i,loc[0])
                cluster[int(loc[0])].append(weight[i])

            self.cluster = cluster

    def get_bmu(self):
        square_difference = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index, self.width) , tf.mod(bmu_index,self.width)])

        return bmu_location

    def update_neighbour(self, bmu):
        learning_rate = .1

        sigma = tf.to_float(tf.maximum(self.width, self.height)/2)


        square_difference = tf.square(self.location - bmu)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        neighbour_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2*tf.square(sigma)))

        rate = neighbour_strength * learning_rate
        total_node = self.width * self.height
        rate_stacked = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.input_dimension]) for i in range(total_node)])

        input_weight_difference = tf.subtract(self.input, self.weight)

        weight_difference = tf.multiply(rate_stacked, input_weight_difference)

        weight_new = tf.add(self.weight, weight_difference)

        return tf.assign(self.weight, weight_new), rate_stacked

def get_data():
    data = pd.read_csv('clustering.csv', usecols=['SpecialDay', 'VisitorType', 'Weekend', 'ProductRelated', 'ExitRates'])
    return data

dataset = get_data()

# Deriving Features
dataset['SpecialDay'] = dataset['SpecialDay'].replace(['HIGH', 'NORMAL', 'LOW'], [2, 1, 0])
dataset['VisitorType'] = dataset['VisitorType'].replace(['Returning_Visitor', 'New_Visitor', 'Other'], [2, 1, 0])
dataset['Weekend'] = dataset['Weekend'].replace([True, False], [1, 0])

# print (dataset)

# Normalize and PCA
def apply_pca(dataset):
    scaler = MinMaxScaler()
    pca = PCA(n_components=3)

    data_normal = scaler.fit_transform(dataset)
    data_pca = pca.fit_transform(data_normal)

    principaldf = pd.DataFrame(data = data_pca, columns= ['PC 1', 'PC 2', 'PC 3'])
    return principaldf

principalComponents = apply_pca(dataset)
# Turn into list
PClist = principalComponents.values.tolist()
# print(principalComponents)


height = 3
width = 3

input_dim = 3
epoch = 5000
som = SOM(height, width, input_dim)

som.train (PClist, epoch)
plt.imshow(som.cluster)
plt.show()







