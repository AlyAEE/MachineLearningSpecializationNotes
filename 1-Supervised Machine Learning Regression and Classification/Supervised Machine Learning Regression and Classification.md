# Supervised Machine Learning: Regression and Classification - Machine Learning specialization

## Machine Learning

- Field of study that gives computers the ability to learn without being explicitly programmed

## Machine Learning Algorithms

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

## Supervised learning

- **Supervised learning** refers to algorithms that learn x→y aka input to output mappings.
- The key characteristic of supervised learning is that you give your learning algorithm examples to learn from. That includes the right answers.
- It’s by seeing correct pairs of input x and label y, supervised learning models learns to take just the input alone and give a reasonable output y.
- the two Major types of **supervised learning** are **Regression** and **Classification**

![images/1.jpg](images/1.jpg)

## Supervised Learning: Regression

- Let’s dive deeper into a specific example: Housing price prediction.

![images/2.jpg](images/2.jpg)

- Here, the x (input) value is the House size in feet squared and the y (output) is the Price in $1000’s.
- **So how can the learning algorithm help us predict house values?**
- One thing it can do is fit a **straight**, or in this case, a **curved** line that fits the data distribution well.
- Furthermore, this housing price prediction problem is a specific type of **supervised learning** known as **regression**.
- **Regression**: Predict a number from infinitely possible outputs.

## Linear Regression

- Linear Regression is one example of a regression models.
- it Fits a straight line through the dataset like the example below:

![images/5.jpg](images/5.jpg)

- This is a **regression** model because it predicts numbers, specifically house prices per size.

![Untitled](images/Untitled.png)

![images/6.jpg](images/6.jpg)

Above we can see the breakdown of the life cycle of how a model works. 

- We start with a **training set** (with features and targets) that is fed into the learning algorithm.
- The learning algorithm then produces some **function f**, which is also called **hypothesis** in the Stanford ML lectures. This function, which is also the actual model, is fed the features and returns the prediction for the output y.

![Untitled](images/Untitled%201.png)

- in machine learning ************w************ and ****b parameters**** are variables that you can adjust during training in order to improve the model. They might be called **Coefficients** or **weights.**
- you can see above how the values of ************w************ and ****b****  differs how the line looks
- What if our training set had **multiple features** as input parameters?
    - Here we take the features **w** as a row vector.
    - **b** is a single number.
    - **Multiple linear regression** is the model for multiple input features. The formula is displayed below:
    
    ![images/12.jpg](images/12.jpg)
    

### Cost Function for Linear Regression (Squared Error)

- In order to implement **linear regression**, we need to first define the cost function.
- The **cost function** tells us how well the model is doing, so we can improve it further.
    - Depending on what the values of **w**, **b** are, our function changes.
- The **cost function** is essentially trying to find **w**, **b** such that the predicted value of **y-hat** is as close to the actual value for **y** for all the values of (**x, y**).

![images/7.jpg](images/7.jpg)

- It takes the prediction **y-hat** and compares it to the target **y** by taking **y-hat − y**.
    - This difference is called **error**, aka how far off our prediction is from the target.
- Next, we will compute the square of this error. We will do this because we will want to compute this value from different training examples in the training set.
- Finally, we want to measure the **error** across the entire training set. Thus, we will sum up the **squared error**.
- To build a **cost function** that doesn’t automatically get bigger as the training set size gets larger by convention, we will compute the **average squared error** instead of the total squared error, and we do that by dividing by **m** like this.
- The last part remaining here is that by convention, the cost function that is used in ML divides by **2** times **m**. This extra division by 2 is to make sure our later calculations look neater, but the **cost function** is still effective if this step is disregarded.
- **J(w, b)** is the **cost function** and is also called the **squared error cost function** since we are taking the squared error of these terms.
- The **squared error cost function** is by far the most commonly used cost function for **linear regression**, and for all regression problems at large.
- The **cost function** measures how well a line fits the training data.

![Untitled](images/Untitled%202.png)

- Goal: Find the parameters **w or w, b** that result in the **smallest** possible **value** for the **cost function J**.
- Keep in mind the cost function **J** will not be in a **1D** space, so minimizing this is not an easy task.

![images/8.jpg](images/8.jpg)

- **gradient descent** and its variations are used to train, not just **linear regression**, but other more common models in AI.

### Batch Gradient Descent or Stochastic Gradient Descent

- **Gradient descent** can be used to **minimize** any function, but here we will use it to minimize our **cost function** for **linear regression**.
    - How does **Gradient Descent** minimize **linear regression** **cost function**?
    - Start with some initial guesses for parameters **w, b**.
    - Computes **gradient** using a single Training example.
    - Keep changing the values for **w, b** to reduce the **cost function J(w, b)**.
    - Continue until we settle at or near a minimum. Note, some **functions** may have **more than 1 minimum**.

![images/9.jpg](images/9.jpg)

![Untitled](images/Untitled%203.png)

- the **learning rate alpha**, it determines how big of a **step** you take when updating **w** or **b**.
- If the **learning rate** is too **small**, you end up taking too many steps to hit the **local minimum** which is inefficient. **Gradient descent** will work but it will be too slow.
- If the **learning rate** is too **large**, you may take a step that is too **big** as miss the **minimum**. **Gradient descent** will fail to converge.

![Untitled](images/Untitled%204.png)

![Untitled](images/Untitled%205.png)
- In the image above, The **Gradient descent** can reach a **local minimum** instead of a **global minimum,**  depends on where you initialize your parameters .

![Untitled](images/Untitled%206.png)

- Fortunately, the **cost function** with **linear regression** doesn’t have multiple **minimums**, it only haves one **global minimum**, This function is called **convex function**.
- How should we choose the **learning rate** then? A few good values to start off with are **0.001,0.01,0.1,1** and so on.
- For **each value**, you might just run **gradient descent** for a handful of iterations and plot the **cost function J** as a function of the number of iterations.
- After picking a few values of the learning rate, you may pick the value that seems to **decrease** the **learning rate** rapidly.
- How do we know we are close to the **local minimum**? **Gradient Descent** can reach a **local minimum** with a **fixed learning rate** because the **derivative** gets closer to **zero** when we approach the **minimum**, so eventually we take smaller steps until we finally reach a **local minimum.**
- How can we check **gradient descent** is working correctly?
    - We can have **2** ways to achieve this. We can plot the cost function **J**, which is calculated on the training set, and plot the value of **J** at each iteration (aka each simultaneous update of parameters **w, b**) of gradient descent.
    - We can also use an **Automatic convergence test**. We choose an **ϵ** to be a very small number. If the cost **J** decreases by less than **ϵ** on one iteration, then you’re likely on this **flattened** part of the curve, and you can **declare convergence**.
    
    ![images/13.jpg](images/13.jpg)
    

### Adam Algorithm (Adaptive Moment Estimation)

- Adam algorithm can see if our learning rate is too small and we are just taking tiny little steps in a similar direction over and over again. It will make the learning rate larger in this case.
- On the other hand, if our learning rate is too big, where we are oscillating back and forth with each step we take, the Adam algorithm can automatically shrink the learning rate.
- Adam algorithm can adjust the learning rate automatically. It uses different, unique learning rates for all of your parameters instead of a single global learning rate.

![Untitled](images/Untitled%207.png)

- Adam algorithm can help you train neural networks faster than gradient descent
- Below is the code for the Adam algorithm:
    
    ![images/22.jpg](images/22.jpg)
    

### Batch Gradient Descent

- **Batch Gradient Descent** is a type of **Gradient Descent** algorithm that at each gradient descent step we look at all the training examples instead of a subset of the data.
- **Batch gradient descent** is an expensive operation since it involves calculations over the full training set at each step. However, if the function is **convex** or relatively smooth, this is the best option.
- **Batch G.D.** also scales very well with a **large number** of **features**.
- Computes gradient using the entire Training sample.
- Gives **optimal solution** if its given sufficient time to **converge** and it will not escape the **shallow local minima** easily, whereas **S.G.D.** can.

### Feature Scaling

![Untitled](images/Untitled%208.png)

- **feature scaling** a techniques that make gradient descent work much better .
- This will enable **gradient decent** to run much faster.
- What **feature scaling** does is that it makes the features involved in the gradient descent computation are all on the similar scale.
- This ensures that the gradient descent moves smoothly towards the **minima** and that the steps for gradient descent are updated at the same rate for all the features.
- Having features on a **similar scale** helps gradient descent converge more quickly towards the **minima**.
- There are many ways to apply **Feature scaling**:

![Untitled](images/Untitled%209.png)

![Untitled](images/Untitled%2010.png)

![Untitled](images/Untitled%2011.png)

## Feature Engineering

![Untitled](images/Untitled%2012.png)

## Cost Function for Linear Regression Via Gradient Descent

![images/10.jpg](images/10.jpg)

## Gradient Descent Derivatives for multiple Linear Regression

![Untitled](images/Untitled%2013.png)

## Normal Equation

- An alternative for gradient descent, that works only for linear regression that solves for w, b without iterations
- Also training becomes slow when the no. of features is large(>10000)
- some machine learning libraries might use Normal Equation to implement Linear Regression

### Polynomial Regression

- We’ve seen the idea of fitting a straight line to our data with linear regression, now lets take a look at **polynomial regression** which will allow you to fit **curves** and **non linear** functions to your data.

![images/14.jpg](images/14.jpg)

## Supervised Learning: Classification

- There's a second major type of **supervised learning** algorithm called a **classification** algorithm
- **Classification** is different from **Regression** because it predicts **categories** not numbers where there are only a small number of possible outputs.
- Let’s dive deeper into a specific example: breast cancer detection prediction.

![images/3.jpg](images/3.jpg)

- You want to classify the tumors as **benign(0)** or **malignant(1)**
- Why is this a **classification** problem? Because the output here is a definitive prediction of a class (aka category), either the cancer is **malignant** or it is **benign**.
- This is a **binary classification**, however, your model can return more **categories** like the image above. It can return different/specific types of malignant tumors.
- **Classification** predicts a small finite number of possible outputs or categories, but not all possible categories in between like **regression** does.
    
    ![images/4.jpg](images/4.jpg)
    
- you can have two or more **inputs** given to the model as displayed above.
- In this scenario, our learning algorithm might try to find some **boundary** between the **malignant** tumor vs the **benign** one in order to make an accurate prediction.
- The learning algorithm has to decide how to fit the **boundary** line.
- **Linear regression** algorithm does not work well for classification problems. We can look at the image below, in the case of an **outlier x**, **linear regression** will have a worse fit to the data set and move its **decision boundary** to the right. The blue vertical decision boundary is supposed to be the decision between malignant and non-malignant.
- However, once we add the **outlier** data, the **decision boundary** changes and thus, the classification between malignant and non-malignant changes, thus giving us a much worse learned function.
    
    ![images/15.jpg](images/15.jpg)
    

### Logistic Regression

- The solution to our classification problem, it’s used for classification problems because it returns a value between **0** or **1.**
- To build out the **logistic regression** algorithm, we use the **Sigmoid function** because it is able to return us an output between **0** and **1**.
    - Lets look at its representation in the image below:
        
        ![images/16.jpg](images/16.jpg)
        
        We can think of the output as a “**probability**” which tells us which class the output maps to.
        
- So if our output is between **0** and **1**, how do we know which class (malignant or not-malignant) the output maps to?
    - The common answer is to pick a **threshold** of **0.5**, which will serve as our **decision boundary**.

![Untitled](images/Untitled%2014.png)

- We can also have **non-linear decision boundarie**s like seen in the image below:
    
    ![images/17.jpg](images/17.jpg)
    

### Gradient Descent for Logistic Regression

- Remember this is the algorithm to minimize the cost function.
- One thing to note here is that even though the formula for gradient descent might look the same as it did when we were looking at linear regression, the function f has changed:
    
    ![images/18.jpg](images/18.jpg)
    

### Cost Function for Logistic Regression (Log Loss)

- One thing to keep in mind is **Loss** and **Cost function** mean the same thing and are used interchangeably.
- Below lets look at the function for the Logistic loss function. Remember that the loss function gives you a way to measure how well a specific set of parameters fits the training data. Thereby gives you a way to try to choose better parameters.
    
    ![images/19.jpg](images/19.jpg)
    

![Untitled](images/Untitled%2015.png)

- Note **y** is either **0** or **1** because we are looking at a binary classification in the image below. This is a simplified loss function.
    
    ![images/20.jpg](images/20.jpg)
    
- Even though the **formula’s** are different, the **baseline** of **gradient descent** are the same. We will be monitoring gradient descent with the l**earning curve and implementing with vectorization**.

## Overfitting and Underfitting

- **Overfitting** is a problem we see when our data “**memorizes**” the **training data**. It will perform very well for training data but when it will **not generalize** well (not have the same accuracy for the validation set). The **cost function** might actually be **= 0** in this case. The model here has **high variance**.
    - Viable solutions for Overfitting:
        - **Collect** more **training** examples. With a larger training set, the learning function will be able to fit the dataset better.
        - **Select features**. See if you can use fewer or more features depending on your current model. If you have a lot of features and insufficient data, you may result in overfitting. Instead, you can select a subset of those features to use given the disadvantage of loosing some useful features.
        - **Regularization**: reduce the size of parameters. Gently reduces the impact of certain features. Lets you keep all your features but doesn’t allow any one feature to have a large impact.
            - Keep in mind, you don’t need to regularize b, usually just regularize w
- **Underfitting** is when we have **high bias** and the data does not fit the training data set well.
    - Viable solutions for underfitting:
        - Get more **training data**.
        - **Increase** the **size** or number of parameters in the model.
        - **Increase** the **complexity** of the model.
        - **Increasing** the **training** time, until **cost function** is minimized.
    
    ![images/21.jpg](images/21.jpg)
    

## Regularization

![Untitled](images/Untitled%2016.png)

- In this modified **cost function**, we want to **minimize** the **original cost**, which is the **mean squared error** plus additionally the second term which is called the **regularization term.**
- This new **cost function** **trades off two goal**s that you might have. Trying to **minimize** this **first** term encourages the algorithm to fit the training data well by **minimizing** the **squared differences** of the predictions and the actual values. And try to **minimize** the second term. The algorithm also tries to keep the parameters **wj small**, which will tend to reduce **overfitting**. The value of **lambda** that you choose, specifies the relative balance between these two **goals**.

## Implementing gradient descent with Regularization

![Untitled](images/Untitled%2017.png)

![Untitled](images/Untitled%2018.png)

![Untitled](images/Untitled%2019.png)

## TL;DR

- Supervised learning maps x the input to the output y and learns from being given the “right answers”. This is also called training your model.
- The two major types of supervised learning algorithms are regression and classification.
- Regression: Predict a number from infinitely many possible outputs.
    - Linear Regression is a type of regression where the model’s cost function is fit with a linear line.
    - The model is represented by a function, with parameters (w,b), and the cost function’s goal is to minimize J(w,b) to find the best fitting line through the data set.
- Classification: Predict a category for a finite number of possible outputs.