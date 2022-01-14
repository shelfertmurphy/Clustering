This text file is a description of my KMeans example and mathematical explanation of the algorithm.

My KMeans example takes a standard RGB image and uses K-Means clustering to compress the image and convert to black and white. I create a KMeans class which can now be reused for other examples.

The K-Mean algorithm takes data points (vectors) and groups the data points into K clusters by minimizing the distance of all data points to the closest centroid (center of a cluster). If we fix a centroid ( 𝜇𝑗 ) and minimize the distance from a data point  𝑥𝑖  to that centroid, our value of  𝑟𝑖𝑗  will return 1 becasue the one indicates that our data point  𝑥𝑖  will be a member of that cluster. Otherwise, we return a 0. Then, we can fix the clusters  𝑟𝑖𝑗  and minimize over the centroid, we will find that the centroids can be computed by  𝑢𝑗=∑rij*x/rij.  This is because we are finding the centroids that minimize the total distance between points and clusters when compared to other groupings.

All values of K produce a black and white image, but as K increases the variety of shades in between black and white increases. We get more shades of grey.