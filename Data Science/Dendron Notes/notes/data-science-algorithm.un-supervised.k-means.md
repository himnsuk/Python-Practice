---
id: qlddebsyddtfgmi97k3bvkd
title: K Means
desc: ''
updated: 1667378747781
created: 1667378747781
---


### Elbow Method

In the Elbow method, we are actually varying the number of clusters ( K ) from 1 – 10. For each value of K, we are calculating **WCSS ( Within-Cluster Sum of Square )**. **WCSS is the sum of squared distance between each point and the centroid in a cluster**. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1. When we analyze the graph we can see that the graph will rapidly change at a point and thus creating an elbow shape. From this point, the graph starts to move almost parallel to the X-axis. The K value corresponding to this point is the optimal K value or an optimal number of clusters.


![Elbow Method](assets/images/data-science-algos/un-supervised/k-means/2022-11-02-14-16-26.png)
