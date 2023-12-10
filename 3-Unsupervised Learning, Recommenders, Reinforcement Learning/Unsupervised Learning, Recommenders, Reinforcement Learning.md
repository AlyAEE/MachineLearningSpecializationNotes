# Unsupervised Learning, Recommenders, Reinforcement Learning

## Unsupervised Learning

- After **supervised learning**, the most common form of machine learning is **unsupervised learning**. In **unsupervised** learning, we are given data without any output **labels y**.
- Data comes with **inputs x** but no **outputs y** and the algorithm has to find structure in this data.
- **Unsupervised** learning model is to find the some structure, some pattern or something interesting in the data.

![Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/1.jpg](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/1.jpg)

- We’re not asked here to predict whether the tumor is malignant or benign because we are not given any labels of which tumor is which.
- Instead, our job is to find some pattern, or some data, or just something interesting within this unlabeled dataset.
- The reason this is called unsupervised learning is that we are not asking the algorithm to give us a “right answer”.
- In this example, our unsupervised algorithm might decide there are two clusters, with one group here and one there.
- This is a specific type of **unsupervised learning** algorithm called **clustering** algorithm because it places the unlabeled data into different clusters.

## Unsupervised learning: Clustering

- **Clustering** groups similar data points together.
- **Clustering** has many use cases:
    - It is used in Google News! Google News looks at 100’s of stories every day and clusters them together.
    
    ![Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/2.jpg](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/2.jpg)
    
    - It is used in DNA microarray clustering. The red here might represent a gene that affects eye color, or the green here is a gene that affects how tall someone is.
        - You can run a clustering algorithm to group different types of individuals together based on categories the algorithm has automatically decided.
    - It is used in grouping customers in different market segments to better understand a company’s consumer base. This could help in improving marketing strategies for each group.

### K-means Algorithm

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled.png)

- The first thing that K-means does is it will take a random guess at where might be the centers of the clusters that you ask it to find.
- It will randomly pick two points at where might be the centers of the two different clusters, This is just a random initial guess and they’re not particularly good guesses.
- Then, **K-means** will repeatedly do two different things. The first is assign points to cluster centroids and the second is move **cluster centroids**.

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%201.png)

- **K-means** will go through all training examples, and for each of them it will check if it closer to the red or blue cluster **centroid**.
- Then, it will assign each point to its closest **centroid**.

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%202.png)

- Then, The algorithm will look at all of the red points and take the average of them, and move the red cross to that average location and do the same thing with blue cross.

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%203.png)

- After repeating these two steps, You will find that there are no more changes to the colors of the points or the location of the **clusters centroids**.
- This means that at this point the **K-means algorithms** has converged

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%204.png)

- $\mu$ are vectors that have the same dimensions as your training examples
- What happens if a cluster has zero training examples assigned to it?
    - The most common thing to do is just eliminate that cluster, you will end up with $k-1$  clusters
    - If you need $k$ clusters, you can randomly reinitialize that cluster centroid  and hope it gets assigned at some points.

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%205.png)

- **K-means** is also optimizing a specific **cost function**

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%206.png)

- The **cost function** for **K-means** is the average **squared distance** between every training example $x^{(i)}$ and the location of the **cluster centroid** to which the training example $x^{(i)}$ has been assigned .
- It turns out that what the **K-means** algorithm is trying to find assignments of points of **clusters** **centroid** as well as find locations of **clusters** centroid that **minimizes** the **squared** distance.
- The fact that **K-means** algorithm is optimizing a **cost function** $J$ means that it is guaranteed to converge, that is on every single iteration. The **distortion cost function** should go down or stay the same.
- How to apply **Random initialization(step 1)** in **k-means**?
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%207.png)
    
    - When running **K-means**, you should  always choose the number of **cluster centroids $K$** to be less than the **training examples $m$**
    - In order to choose the **cluster centroids**, the most common way is to randomly pick **K training examples**.
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%208.png)
    
    - Depending on how you choose the random initial **cluster centroids K-means** will end up picking a difference set of **clusters** for your data set.
    - You might end up with **clustering** which doesn’t look as good, this turns out to be a **local optima**, in which **k-means** tries to minimize the **distortion cost** function $j$ , but with less unfortunate **random initialization** , it happened to get stuck in a **local minimum.**
    - In order for **k-means** to find the **global** minimum, is to run it multiple times and choose the one with the lowest value for the **cost** function $j$   ****
    - Here’s the algorithm for **********************random initialization:**********************
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%209.png)
    
- How to choose the number of **clusters** to use ****for **k-means**?
    - For a lot of clustering problems, the right value of $**k**$ is truly ambiguous.
    - There are a few techniques to try to automatically choose the number of **clusters** to use for a certain application.
        1. The ************elbow************ method(not recommended):
            1. you would run **k-means** with variety of $k$ values and plot the **cost function** $j$ as a function of the no. of clusters and pick the value that look like an **elbow**.
        
        ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2010.png)
        
1. Evaluate $k$ based on how well it performs on the downstream purpose
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2011.png)
    
    a. In case of running a T-shirt business, both of the solution are fine, but whether you want to use 3 or 5 clusters depends on what make sense for your T-shirt business.
    
    b. There’s a trade-off between how well the T-shirts will fit (3 or 5 sizes) or the extra costs associated with manufacturing and shipping five types of T-shirts instead of three. 
    

### Anomaly Detection Algorithm

- **Anomaly Detection** is another specific type of unsupervised learning and it is used to detect unusual events or data points
- **Anomaly detection** algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or an anomalous event.

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2012.png)

- How can you have an algorithm address this problem?
    - The most common way to carry out **anomaly detection** is through a technique called **density estimation**
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2013.png)
    
    - When you're given your training sets of **m** examples, the first thing you do is build a model for the probability of **x**.
    - The learning algorithm will try to figure out what are the values of the features x1 and x2 that have high probability and what are the values that are less likely or have a lower chance or lower probability of being seen in the data set.
    - When you are given the new test example $**Xtest**$. you will its compute the probability, And if it is less than some small number **epsilon**, we will raise a flag to say that this could be an anomaly.
- **Anomaly Detection** is used in:
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2014.png)
    
- In order to apply **anomaly detection**, we're going to need to use the **Gaussian distribution(normal distribution**).

![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2015.png)

- what is the **Gaussian** or the **normal distribution?**
    - Say **x** is a number, and if x is a random number, sometimes called the random variable. If the probability of **x** is given by a **Gaussian** or **normal distribution,** this means the probability of **x** looks like this curve in the image above.
    - The center or the middle of the curve is given by the **mean** $\mu$, and the **standard deviation** or the width of this curve is given by that **variance** parameter **Sigma**.
    - For any given value of $\mu$  and $\sigma$ , if you were to plot this function $p(x)$  , you get this type of **bell-shaped curve** that is centered at $\mu$, and with the width of this **bell**-**shaped curve** being determined by the parameter $\sigma$.
- How changing **mean** $\mu$ and **sigma** $\sigma$ will affect the **gaussian distribution**?
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2016.png)
    
- What is ************maximum likelihood?************
    - In statistics, **maximum likelihood estimation** (**MLE**) is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by **maximizing** a **likelihood** function so that, under the assumed statistical model, the observed data is most probable.
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2017.png)
    
    - If you set $\mu$ according to this formula above and $\sigma ^{2}$ according to this formula, you’d get a **Gaussian distribution** that will be a possible probability distribution in terms of what's the probability distribution that the training examples had come from.
- How to implement **anomaly detection**?
    1. Apply **Density Estimation**
        1. Estimate the **probability** of any given **feature vector** $p(\vec x)$
            1. In statistics, This equation assumes that these features are statistically independent, but it turns out, anomaly detection works even if the features are not statistically independent.
        
        ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2018.png)
        
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2019.png)
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2020.png)
    
- How to evaluate an anomaly detection algorithm?
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2021.png)
    
    - If you have a way to evaluate a system, even as it's being developed, you'll be able to make decisions and change the system and improve it much more quickly.
    - **Real number evaluation** means that if you can quickly change the algorithm in some way, such as change a feature or change a parameter and have a way of computing a number that tells you if the algorithm got better or worse, then it makes it much easier to decide whether or not to stick with that change to the algorithm. This is how it's often done in **anomaly detection**.
    
    ---
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2022.png)
    
    - Even though we've mainly been talking about **unlabeled** data, assume that we have some labeled data, of some previously observed **anomalies**.
    - To evaluate your algorithm, come up with a way to have a **real number evaluation**, it turns out to be very useful if you have a small number of **anomalous** examples included in a **cross validation set** and a **test set**.
    - In practice, **anomaly detection** algorithm will work okay if there are some examples that are actually **anomalous**, but there were accidentally labeled with **y** equals **0**.
    - What you can do then is train the algorithm on the **training set**, fit the **Gaussian distributions** to these 6,000 examples and then on the **cross-validation set**, you can see how many of the **anomalous** engines it correctly flags.
    - you could use the **cross-validation set** to tune the parameter **epsilon** and set it higher or lower depending on whether the algorithm seems to be reliably detecting these 10 anomalies without taking too many of these 2,000 good engines and flagging them as **anomalies**.
    - After you have tuned the parameter **epsilon** and applied **feature engineering,** you can then take the algorithm and evaluate it on your **test set** to see how many of these 10 anomalous engines it finds, as well as how many mistakes it makes by flagging the good engines as **anomalous** ones.
    - If you're building a practical **anomaly detection** system, having a small number of **anomalies** to use to evaluate the algorithm that your **cross-validation** and **test sets** is very helpful for tuning the algorithm.
    - There’s also another alternative to not use a **test set**, If you have a few anomalous examples, like to have just a **training set** and a **cross-validation set**.
    - The downside of this alternative here is that after you've tuned your algorithm, you don't have a fair way to tell how well this will actually do on future examples because you don't have the **test set**. Just be aware that there's a higher risk that you will have **over-fit** some of your decisions around **Epsilon** and choice of **features** and so on to the **cross-validation set** and so its performance on real data in the future may not be as good as you were expecting.
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2023.png)
    
    - To evaluate the algorithm on your **cross-validation sets** or on the **test set**. You would first fit the model $p(\vec x)$  on the **training** **set**. Then on any **cross-validation** or **test set.**
    - Based on the formula above, you can now look at how accurately this algorithm's predictions on the **cross-validation** or **test set** matches the **labels** **y** you have in the **cross-validation** or the **test sets**.
    - Since most of **anomaly** detection application datasets are **skewed** datasets, there are different metrics to evaluate the **cross-validation** and **test sets**
    - In conclusion, the practical process of building an **anomaly detection** system is much easier if you actually have just a small number of **labeled examples** of known **anomalies**.
- Now, this does raise the question, if you have a few labeled examples, since you'll still be using an unsupervised learning algorithm, why not take those labeled examples and use a supervised learning algorithm instead?
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2024.png)
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2025.png)
    
- How to effectively choose a good set of features when building an **anomaly detection algorithm**?
    - 
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2026.png)
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2027.png)
    
    ![Untitled](Unsupervised%20Learning,%20Recommenders,%20Reinforcement%2082c1b9697502486782fb343011c83abc/Untitled%2028.png)