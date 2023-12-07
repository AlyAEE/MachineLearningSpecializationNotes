# Unsupervised Learning, Recommenders, Reinforcement Learning

## Unsupervised Learning

- After **supervised learning**, the most common form of machine learning is **unsupervised learning**. In **unsupervised** learning, we are given data without any output **labels y**.
- Data comes with **inputs x** but no **outputs y** and the algorithm has to find structure in this data.
- **Unsupervised** learning model is to find the some structure, some pattern or something interesting in the data.

![images/1.jpg](images/1.jpg)

- We’re not asked here to predict whether the tumor is malignant or benign because we are not given any labels of which tumor is which.
- Instead, our job is to find some pattern, or some data, or just something interesting within this unlabeled dataset.
- The reason this is called unsupervised learning is that we are not asking the algorithm to give us a “right answer”.
- In this example, our unsupervised algorithm might decide there are two clusters, with one group here and one there.
- This is a specific type of **unsupervised learning** algorithm called **clustering** algorithm because it places the unlabeled data into different clusters.

## Unsupervised learning: Clustering

- **Clustering** groups similar data points together.
- **Clustering** has many use cases:
    - It is used in Google News! Google News looks at 100’s of stories every day and clusters them together.
    
    ![images/2.jpg](images/2.jpg)
    
    - It is used in DNA microarray clustering. The red here might represent a gene that affects eye color, or the green here is a gene that affects how tall someone is.
        - You can run a clustering algorithm to group different types of individuals together based on categories the algorithm has automatically decided.
    - It is used in grouping customers in different market segments to better understand a company’s consumer base. This could help in improving marketing strategies for each group.

### K-means Algorithm

![Untitled](images/Untitled.png)

- The first thing that K-means does is it will take a random guess at where might be the centers of the clusters that you ask it to find.
- It will randomly pick two points at where might be the centers of the two different clusters, This is just a random initial guess and they’re not particularly good guesses.
- Then, **K-means** will repeatedly do two different things. The first is assign points to cluster centroids and the second is move **cluster centroids**.

![Untitled](images/Untitled%201.png)

- **K-means** will go through all training examples, and for each of them it will check if it closer to the red or blue cluster **centroid**.
- Then, it will assign each point to its closest **centroid**.

![Untitled](images/Untitled%202.png)

- Then, The algorithm will look at all of the red points and take the average of them, and move the red cross to that average location and do the same thing with blue cross.

![Untitled](images/Untitled%203.png)

- After repeating these two steps, You will find that there are no more changes to the colors of the points or the location of the **clusters centroids**.
- This means that at this point the **K-means algorithms** has converged

![Untitled](images/Untitled%204.png)

- $\mu$ are vectors that have the same dimensions as your training examples
- What happens if a cluster has zero training examples assigned to it?
    - The most common thing to do is just eliminate that cluster, you will end up with $k-1$  clusters
    - If you need $k$ clusters, you can randomly reinitialize that cluster centroid  and hope it gets assigned at some points.

![Untitled](images/Untitled%205.png)
- **K-means** is also optimizing a specific **cost function**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/997c02c6-da32-4ad2-b102-4daf806dd0f6/Untitled.png)

- The **cost function** for **K-means** is the average **squared distance** between every training example $x^{(i)}$ and the location of the **cluster centroid** to which the training example $x^{(i)}$ has been assigned .
- It turns out that what the **K-means** algorithm is trying to find assignments of points of **clusters** **centroid** as well as find locations of **clusters** centroid that **minimizes** the **squared** distance.
- The fact that **K-means** algorithm is optimizing a **cost function** $J$ means that it is guaranteed to converge, that is on every single iteration. The **distortion cost function** should go down or stay the same.
- How to apply **Random initialization(step 1)** in **k-means**?
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/6c281955-67fe-4c95-b3cf-722e0db6d3c3/Untitled.png)
    
    - When running **K-means**, you should  always choose the number of **cluster centroids $K$** to be less than the **training examples $m$**
    - In order to choose the **cluster centroids**, the most common way is to randomly pick **K training examples**.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/77c0b13b-f013-46ed-bc91-f62011c6ccd9/Untitled.png)
    
    - Depending on how you choose the random initial **cluster centroids K-means** will end up picking a difference set of **clusters** for your data set.
    - You might end up with **clustering** which doesn’t look as good, this turns out to be a **local optima**, in which **k-means** tries to minimize the **distortion cost** function $j$ , but with less unfortunate **random initialization** , it happened to get stuck in a **local minimum.**
    - In order for **k-means** to find the **global** minimum, is to run it multiple times and choose the one with the lowest value for the **cost** function $j$   ****
    - Here’s the algorithm for **********************random initialization:**********************
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/01a2c285-c919-4b66-af67-dd32f486dc98/Untitled.png)
    
- How to choose the number of **clusters** to use ****for **k-means**?
    - For a lot of clustering problems, the right value of $**k**$ is truly ambiguous.
    - There are a few techniques to try to automatically choose the number of **clusters** to use for a certain application.
        1. The ************elbow************ method(not recommended):
            1. you would run **k-means** with variety of $k$ values and plot the **cost function** $j$ as a function of the no. of clusters and pick the value that look like an **elbow**.
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/bbedab86-00da-48c0-9a6b-904309a01aba/Untitled.png)
        
1. Evaluate $k$ based on how well it performs on the downstream purpose
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/8922aaf7-be84-4ccc-8d4e-ef3a7243439a/b624fdd2-eee8-499a-ad6e-caaf0e31bae3/Untitled.png)
    
    a. In case of running a T-shirt business, both of the solution are fine, but whether you want to use 3 or 5 clusters depends on what make sense for your T-shirt business.
    
    b. There’s a trade-off between how well the T-shirts will fit (3 or 5 sizes) or the extra costs associated with manufacturing and shipping five types of T-shirts instead of three.

### Anomaly Detection Algorithm

- **Anomaly Detection** is another specific type of unsupervised learning and it is used to detect unusual events or data points
- Anomaly Detection is used in:
    - **Fraud** Detection

### Dimensionality Reduction

- Compress data using fewer numbers.