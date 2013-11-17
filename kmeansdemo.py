
# Step-by-step demo of the k-means algorithm.
# Partitions a list of numbers (data points) in 2 clusters.
# Parameters:
#   points: a list of numbers to be clustered
#   centroid1: initial centroid #1
#   centroid2: initial centroid #2
# Returns nothing
def kMeansDemo(points, initialCentroid1, initialCentroid2):
    print "K-means clustering demo"
    print "Points =", points
    print "Initial centroid 1 =", initialCentroid1
    print "Initial centroid 2 =", initialCentroid2
    previousCentroid1 = None
    previousCentroid2 = None
    centroid1 = initialCentroid1
    centroid2 = initialCentroid2
    i = 0
    while centroid1 != previousCentroid1 or centroid2 != previousCentroid2:
        unused = raw_input("\nPress any key to continue")
        i += 1
        print "--------------------------------------------------"
        print "Iteration", i
        print "\nLet's compute the distances between all the data points"
        print "and the current centroids:"
        print "\t%.1f\t%.1f" % (centroid1, centroid2)
        cluster1 = []
        cluster2 = []
        for x in points:
            distance1 = abs(centroid1 - x)
            distance2 = abs(centroid2 - x)
            if distance1 < distance2:
                cluster1.append(x)
                stringDistance1 = "(%.1f)" % distance1
                stringDistance2 = "%.1f" % distance2
            else:
                cluster2.append(x)
                stringDistance1 = "%.1f" % distance1
                stringDistance2 = "(%.1f)" % distance2
            print "%.1f\t%s\t%s" % (x, stringDistance1, stringDistance2)
        print "\nLet's assign each point to the cluster represented"
        print "by the closest centroid:"
        print "cluster 1 =", cluster1
        print "cluster 2 =", cluster2
        print "\nLet's compute the new value of the centroids as the mean"
        print "of the points currently in each cluster:"
        previousCentroid1 = centroid1
        previousCentroid2 = centroid2
        centroid1 = sum(cluster1) / float(len(cluster1))
        centroid2 = sum(cluster2) / float(len(cluster2))
        print "centroid 1 = %s = %.1f" % (stringMean(cluster1), centroid1)
        print "centroid 2 = %s = %.1f" % (stringMean(cluster2), centroid2)

        #now let's plot the points and the clusters to which they belong

        # @mahalia this is probably where we'd want to plot the updates
    print "--> The centroids have not changed. We are done!"

# Creates a string representation of the computation
# of the mean of a list of numbers
# Parameters:
#   myList: a list of numbers
# Returns: a string
def stringMean(myList):
    if len(myList) == 0:
        return ""
    else:
        s = "(" + str(myList[0])
        for i in range(1, len(myList)):
            s += " + " + str(myList[i])
        s += ") / " + str(len(myList))
        return s

points = [80, 65, 70, 65, 86, 70, 38, 67, 38, 41, 70, 70, 70, 70, 65]
initialCentroid1 = 59
initialCentroid2 = 84
kMeansDemo(points, initialCentroid1, initialCentroid2)





#ultra dark vs. other: 75,90
#milk vs. dark: 38,41 OR 38, 70


# points = [38, 65, 67, 85, 65, 70, 70, 80, 70, 86, 38, 65, 70, 41, 70, 70]
