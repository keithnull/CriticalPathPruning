from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def test_cluster(centroid, test_images, test_labels):
	d = CifarDataManager()
	model = TrimmedModel(centroid)
	model.assign_weight()
	return model.test_cluster(test_images, test_labels)

if __name__ == "__main__":
	centroids = np.load("./kmeansRes/KMeans-10-centroids.npy")
	labels = np.load("./kmeansRes/KMeans-10-labels.npy")
	
	d = CifarDataManager()
	test_images, test_labels = d.test.next_batch(10)
	
	num_of_cluster = 2
	res = [test_cluster(centroids[i], test_images, test_labels) for i in range(num_of_cluster)]

	for img_num in range(len(test_images)):
		pred = [res[cluster][img_num] for cluster in range(num_of_cluster)]
		print(pred)





		