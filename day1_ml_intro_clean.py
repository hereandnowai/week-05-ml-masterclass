#!/usr/bin/env python
# coding: utf-8

# # Day 1: Machine Learning Fundamentals (9:30 – 11:00)

# # WHAT IS MACHINE LEARNING
# 
# Machine Learning is a subset of AI that allows systems to learn patterns from data and make predictions without being explicitly programmed.
# 
# **Example:**
# Predicting house prices based on past data.

# # What is Artificial Intelligence
# 
# Artificial Intelligence (AI) is a broad field that focuses on building systems capable of performing tasks that typically require human intelligence.
# 
# AI is an umbrella term.
# 
# Machine Learning and Deep Learning are subsets of AI.
# 
# IMPORTANT:
# 
# * AI is NOT rule-based programming
# * Rule-based systems are traditional programming
# * AI includes learning-based systems

# # AI vs ML vs DL (WITH CODE DEMOS)
# 
# ## 3.1 Traditional Programming

# In[12]:


def rule_based(x):
    return x * 2

print(rule_based(5))


# ## 3.2 Machine Learning

# In Machine Learning, we don't write the rules. Instead, we provide the **data** and the **answers (labels)**, and the algorithm finds the **rules** (the pattern) for us. We can then use these discovered rules to make predictions on new, unseen data.
# 
# **Example:** Instead of writing `x * 2` manually, we show the system that `1` becomes `2`, `2` becomes `4`, and `3` becomes `6`. The model "learns" that the rule is multiplication by 2.

# In[21]:


import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

print(model.predict(np.array([[5]])))


# In the above code, the model learned a **Linear Relationship** ($y = mx + b$) between $X$ and $y$. 
# 
# 1.  **Learned Slope ($m$):** The model discovered that for every increase of 1 in $X$, $y$ increases by 2.
# 2.  **Learned Intercept ($b$):** The model found that when $X$ is 0, $y$ is also 0.
# 
# Because the model found this straight-line pattern ($y = 2x + 0$), it can now accurately predict that when $X = 5$, $y$ should be **10**, even though it never saw that specific number during training.

# ## 3.3 Deep Learning

# https://playground.tensorflow.org/

# Deep Learning is a subset of Machine Learning that uses **Artificial Neural Networks** (modeled after the human brain). While ML often requires manual data processing, DL can learn complex, hierarchical patterns on its own with multiple **layers** (hence "Deep").
# 
# **Example:** A simple neural network with multiple neurons (`Dense(4)`) can also learn basic patterns (like multiplication by 2) by adjusting "weights" during training.

# In[22]:


import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path}")
try:
    import tensorflow
    print(f"TF Version: {tensorflow.__version__}")
except ImportError as e:
    print(f"Error: {e}")


# In[33]:


import numpy as np
import tensorflow as tf

# Simplified model to find the linear rule y = 2x
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1) # Single neuron is enough for y = mx + b
])

model.compile(optimizer='adam', loss='mse')

X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Increased epochs to 5000 to ensure the loss gets near 0
model.fit(X, y, epochs=5000, verbose=0) 

print(f"Prediction for X=5: {model.predict(np.array([[5]], dtype=float))}")


# ### Simple Breakdown of the Deep Learning Code:
# 
# 1.  **The Brain (Model):** We create a `Sequential` model.
#     *   `Input(shape=(1,))`: Expects one number.
#     *   `Dense(1)`: A single neuron is perfect for learning a line like $y = 2x$. Using too many neurons for a simple problem can sometimes make learning slower because the "brain" is overthinking a simple rule.
# 2.  **The Learning Strategy (Compile):** We use `adam` and `mse`.
# 3.  **The Data:** $X$ ($1, 2, 3, 4$) and $y$ ($2, 4, 6, 8$).
# 4.  **The Training (Fit):** We increased the training to **5000 epochs**.
#     *   **Why so many?** The `Adam` optimizer is designed for complex problems with millions of data points. For a tiny dataset with only 4 rows, it takes many, many small steps to adjust the weights precisely enough to reach exactly 10.0.
# 5.  **The Prediction:** With 5000 rounds of practice, the model should now be extremely close to **10**.

# ## 3.4 Explanation
# 
# * **Traditional Programming** → rules written manually
# * **Machine Learning** → learns patterns from data
# * **Deep Learning** → neural networks with multiple layers

# # REAL-WORLD APPLICATIONS
# 
# Examples of Machine Learning:
# 
# * **Netflix** → Recommendation systems
# * **Amazon** → Product recommendations
# * **Google** → Search ranking
# * **Banking** → Fraud detection
# * **Healthcare** → Disease prediction

# In[41]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

print("Libraries loaded successfully.")


# ## Cosine Similarity in Practice
# Let's build a small genre-preference matrix for 4 users.

# In[42]:


data = {
    'Action': [5, 4, 1, 1],
    'Sci-Fi': [4, 5, 0, 1],
    'Romance': [1, 0, 5, 4],
    'Comedy': [2, 1, 4, 5]
}
users = ['Zenil', 'Mihir', 'Avishka', 'Aditi']
df_ratings = pd.DataFrame(data, index=users)

similarity = cosine_similarity(df_ratings)
df_sim = pd.DataFrame(similarity, index=users, columns=users)

plt.figure(figsize=(8, 6))
sns.heatmap(df_sim, annot=True, cmap='coolwarm')
plt.title("Recommendation Matrix (User Similarity)")
plt.show()


# ### **Understanding the Recommendation Matrix**
# 
# #### **1. What is Cosine Similarity?**
# In simple terms, Cosine Similarity measures the **"Angle"** between two users' preferences. 
# *   If two users like the exact same genres, the angle between them is **0°**, and the similarity score is **1.0** (Perfect Match).
# *   If their tastes are completely opposite, the score is **0.0**.
# 
# #### **2. How to Read This Heatmap:**
# *   **The Diagonal (1.0s):** You’ll notice a diagonal line of "1"s from the top-left to bottom-right. This is because every user is 100% similar to themselves (Zenil vs. Zenil = 1.0).
# *   **Dark Red Cells:** These indicate high similarity. For example, **Zenil and Mihir** have a score of **0.96**. This means they have almost identical tastes in Action and Sci-Fi.
# *   **Blue Cells:** These indicate low similarity. **Mihir and Avishka** have a score of only **0.19**, because Mihir likes Action/Sci-Fi while Avishka prefers Romance/Comedy.
# 
# #### **3. The "Netflix" Logic (Recommendation):**
# Imagine Zenil just watched a new Sci-Fi movie and gave it 5 stars. Because the AI knows **Mihir** is 96% similar to Zenil, it will immediately **recommend** that same movie to Mihir!
# 
# This is the "Math" behind your Netflix home screen. It doesn't just look at what *you* watch; it looks at what "Users like you" are watching.

# # TYPES OF MACHINE LEARNING

# ## 6.1 SUPERVISED LEARNING
# Model learns from labeled data.

# In Supervised Learning, the model is trained on **labeled data**, where each input is paired with its correct output. The model's goal is to learn a mapping function from input to output.
# 
# **Example:** Providing ages (input) and a decision like "buy" or "not buy" (label) to teach the model to classify future customers based on their age.

# In[44]:


from sklearn.tree import DecisionTreeClassifier

X = [[25], [30], [45]]
y = ["buy", "buy", "not buy"]

model = DecisionTreeClassifier()
model.fit(X, y)

print(model.predict([[35]]))


# ### **Interpretation of Supervised Learning**
# In the code above (Cell 28), we use a **Decision Tree**.
# *   **The Input ($X$):** Ages of customers (25, 30, 45).
# *   **The Label ($y$):** Explicit answers ("buy", "not buy").
# *   **The Prediction:** When we ask the model about a 35-year-old, it looks at its training data and decides the most likely category based on what it learned from the labeled examples. It's like a student learning from a textbook with an answer key.

# ## 6.2 UNSUPERVISED LEARNING
# Model finds patterns without labels.

# In Unsupervised Learning, the model looks for **hidden patterns** in the data **without any predefined labels** (answers). It groups similar data points together based on their shared characteristics.
# 
# **Example:** K-Means clustering can group customers based on their spending habits even without being told which customers browse similar items.

# In[9]:


from sklearn.cluster import KMeans

X = [[1], [2], [10], [11]]

model = KMeans(n_clusters=2)
model.fit(X)

print(model.labels_)


# ### **Interpretation of Unsupervised Learning**
# In the code above (Cell 31), we use **K-Means Clustering**.
# *   **The Observations ($X$):** We only provide numbers (1, 2, 10, 11). Note there are **no labels** ($y$) like "small" or "large".
# *   **The Logic:** The model notices that 1 and 2 are close together, while 10 and 11 are close together.
# *   **The Result:** It automatically creates two groups (Clusters). It doesn't know *what* the groups are, just that the items inside them look similar. It's like a baby sorting blocks by color without knowing the names of the colors.

# ## 6.3 SEMI-SUPERVISED LEARNING
# Uses both labeled and unlabeled data.

# Semi-Supervised Learning uses a **small amount of labeled data** and a **large amount of unlabeled data**. The model uses the labeled data to understand the categories and then assigns labels (pseudo-coloring) based on similarities in the unlabeled data.
# 
# **Example:** Labeling a few photos manually (like "dog" or "cat") and letting the algorithm use these examples to categorize thousands of unlabeled images. (In the code below, `-1` represents unlabeled data).

# In[55]:


from sklearn.semi_supervised import LabelSpreading

X = [[1], [2], [3], [10], [11]]
y = [0, 0, -1, 1, -1]

model = LabelSpreading()
model.fit(X, y)

print(model.transduction_)


# ### **Interpretation of Semi-Supervised Learning**
# In the code above (Cell 34), we use **Label Spreading**.
# *   **The Data:** We have 5 points, but we only know the labels for 3 of them. The `-1` represents "Unlabeled".
# *   **The Logic:** The model takes the known labels ($0$ and $1$) and "spreads" them to the unknown neighbors based on proximity.
# *   **The Benefit:** This is how Google Photos works—you label 2 photos of your dog, and it automatically finds the other 500 unlabeled ones that look like him.

# ## 6.4 REINFORCEMENT LEARNING
# Learns through rewards and penalties.

# In Reinforcement Learning, the model (**agent**) learns by **interacting** with an **environment**. It makes **actions** and receives either a **reward** (positive) or a **penalty** (negative) based on the outcome. The goal is to maximize the cumulative reward.
# 
# **Example:** Teaching a computer to play a video game by rewarding it for getting points and penalizing it for dying. (game over).

# In[47]:


import random

for step in range(50):
    action = random.choice(["left", "right"])

    if action == "right":
        reward = 1
    else:
        reward = -1

    print(action, reward)


# ### **Interpretation of Reinforcement Learning**
# 
# In the code above (Cell 37), we simulate an **Agent** and an **Environment**.
# 
# #### **Why the agent never "learns" to only turn right:**
# If you run the code, you'll see it still picks "left" even after being penalized. This is because **this code is only a simulation of actions and rewards**, not a "Brain."
# *   **Missing Memory:** The agent has no "Q-Table" or "Neural Network" to store the result of its last move.
# *   **Pure Randomness:** The line `random.choice(["left", "right"])` forces the agent to be 100% random regardless of the reward.
# 
# #### **How a real RL Agent would learn:**
# In a real system, we would add a **Policy**. The agent would check: *"Last time I went left, I got -1. Last time I went right, I got +1. Therefore, the probability of going right should now be higher."*
# 
# Over hundreds of steps, a real agent would eventually learn that "Right = Good" and stop going left entirely. 
# 
# **Analogy:** This code is like a person flipping a coin. No matter how many times the coin landing on "Tails" makes them lose money, the coin itself doesn't "learn" to land on "Heads"ads!

# ## **6.5 Real Reinforcement Learning (With Memory)**
# 
# Now, let's build a **Real RL Agent**. This agent has a "Brain" (called a **Q-Table**) where it stores the value of its actions. 
# 
# ### **The Logic:**
# 1.  **Memory:** It starts with 0 knowledge for both 'Left' and 'Right'.
# 2.  **Learning Rate:** It learns from its mistakes and successes.
# 3.  **Exploration:** At first, it tries both sides. As it learns 'Right' is better, it naturally starts choosing 'Right' more often. 
# 4.  **The Result:** Unlike the random simulation above, this agent will eventually **stop going left** because it wants the reward!

# In[49]:


import numpy as np
import random

# 1. Initialize the "Brain" (Q-Table)
# index 0 = left, index 1 = right
q_table = np.zeros(2) 
learning_rate = 0.1

print(f"Starting Knowledge: {q_table} (Left: 0, Right: 0)\n")

# 2. Training Loop (The Agent interacts with the environment)
for episode in range(1, 101):
    # Choose action: 10% chance to explore randomly, 90% chance to use knowledge
    is_exploring = False
    if random.uniform(0, 1) < 0.1:
        action_idx = random.randint(0, 1) # Explore
        is_exploring = True
    else:
        action_idx = np.argmax(q_table) # Use Knowledge

    action_name = "RIGHT" if action_idx == 1 else "LEFT"

    # Environment gives reward
    if action_idx == 1: # Right
        reward = 1
    else: # Left
        reward = -1

    # 3. Update the Brain (Bellman Equation simplified)
    old_value = q_table[action_idx]
    q_table[action_idx] = old_value + learning_rate * (reward - old_value)

    # Print progress every 10 steps to show students the learning
    if episode % 10 == 0 or episode <= 5:
        mode = "(Exploring)" if is_exploring else "(Exploiting Knowledge)"
        print(f"Episode {episode:3}: Agent chose {action_name:5} {mode:25} | Reward: {reward:2} | New Q-Table: {q_table}")

print(f"\nFinal Learned Knowledge: {q_table}")
print(f"Action with highest value: {'Right' if np.argmax(q_table) == 1 else 'Left'}")

# 4. Final Test: The Agent now "knows" what to do!
test_action = 'Right' if np.argmax(q_table) == 1 else 'Left'
print(f"\nAI's Decision after training: {test_action}")


# This code implements **Q-Learning**, which is a fundamental **Reinforcement Learning** algorithm.
# 
# Here is how "AI" is working in this specific snippet:
# 
# ### 1. The "AI Algorithm": Q-Learning
# The core of this AI is a **Q-Table** (the `q_table` array). 
# *   In a complex AI (like a self-driving car), this "table" would be a **Deep Neural Network**.
# *   In this simple example, it's a small memory bank that stores the "expected reward" for each possible action (Left vs. Right).
# 
# ### 2. How the AI "Learns" (The Math)
# The AI uses a simplified version of the **Bellman Equation**:
# `q_table[action_idx] = old_value + learning_rate * (reward - old_value)`
# 
# *   **Reward (The Teacher):** The environment provides feedback (`+1` for Right, `-1` for Left).
# *   **Temporal Difference (The Correction):** The AI calculates the difference between what it *expected* to get and what it *actually* got.
# *   **Update:** It updates its "Brain" (Q-Table) just a little bit (`learning_rate = 0.1`) so it doesn't overreact to a single event but identifies a consistent pattern over time.
# 
# ### 3. "Exploration vs. Exploitation" (The Strategy)
# This is a key AI concept called the **Epsilon-Greedy Strategy**:
# *   **Exploration (10%):** The AI randomly tries things to see if there is a better reward it hasn't found yet.
# *   **Exploitation (90%):** The AI uses its current knowledge (`np.argmax`) to make the best possible decision.
# 
# ### 4. Convergence (The Result)
# By the time the loop hits **Episode 100**, the "Value" stored for **RIGHT** in the Q-Table will be significantly higher than **LEFT**. The AI has effectively "programmed itself" to know that turning Right is the optimal behavior to move to survive or get points.

# In[ ]:


# 7. ML PROBLEM TYPES: CLASSIFICATION VS REGRESSION

In Machine Learning, most problems fall into two main categories based on what we are trying to predict.

### 7.1 Regression (Predicting a Quantity)
Regression is used when the output is a **continuous number** (like price, temperature, or height).
*   **Question:** "How much?" or "How many?"
*   **Example:** Predicting the price of a house ($350,200) or the temperature tomorrow (24.5°C).

### 7.2 Classification (Predicting a Category)
Classification is used when the output is a **label or category** (like "Yes/No", "Cat/Dog").
*   **Question:** "Which category?"
*   **Example:** Identifying if an email is "Spam" or "Not Spam", or if an image is a "Car" or "Truck".

---
**Think of it this way:**
*   If you are predicting a **Score** (0-100) — it's **Regression**.
*   If you are predicting **Pass or Fail** — it's **Classification**.


# # **VIDEO SUPPLEMENT: What is Machine Learning?**
# To help solidify these concepts, watch this quick overview.
# 

# In[ ]:


i en


# In[ ]:


from IPython.display import YouTubeVideo

# Embedding a video about Machine Learning Fundamentals
# Video ID: tWwCK95X6go
YouTubeVideo('tWwCK95X6go', width=800, height=450)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# 1. Regression Data (Predicting a number)
X_reg = np.array([[1], [2], [3], [4], [5]])
y_reg = np.array([10, 20, 30, 40, 50]) # Prices/Scores

# 2. Classification Data (Predicting a category)
X_clf = np.array([[1], [2], [3], [8], [9], [10]])
y_clf = np.array([0, 0, 0, 1, 1, 1]) # 0=Fail, 1=Pass

# Vizualizing the difference
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_reg, y_reg, color='blue')
plt.plot(X_reg, y_reg, color='red')
plt.title("Regression: Predicting a Trend (Number)")
plt.xlabel("Input")
plt.ylabel("Continuous Output")

plt.subplot(1, 2, 2)
plt.scatter(X_clf, y_clf, color='green')
plt.title("Classification: Predicting a Group (Class)")
plt.xlabel("Input")
plt.ylabel("Category (0 or 1)")

plt.tight_layout()
plt.show()


# # 8. ML PIPELINE OVERVIEW
# 
# Building an ML model isn't just about code; it's a step-by-step factory process called a **Pipeline**.
# 
# 1.  **Data Collection:** Gathering raw information (CSV files, Databases, Sensors).
# 2.  **Data Preprocessing:** Cleaning the data. Removing duplicates, fixing missing values, and scaling numbers. **(Crucial Step!)**
# 3.  **Model Training:** Feeding the cleaned data into an algorithm so it can learn the patterns.
# 4.  **Evaluation:** Testing the model on data it has never seen before to see if it's actually accurate.
# 5.  **Deployment:** Putting the model into a real app (like a website or a phone) where users can interact with it.
# 
# ---
# 
# # 9. ML PROJECT LIFECYCLE (END-TO-END)
# 
# The lifecycle is a circular process. We don't just "finish"; we improve.
# 
# 1.  **Business Problem:** Define what you want to solve (e.g., "Reduce customer churn").
# 2.  **Data Acquisition:** Get the right data.
# 3.  **EDA (Exploratory Data Analysis):** Visualize and understand the data.
# 4.  **Modeling:** Choose and train the right AI "Brain".
# 5.  **Deployment:** Launch the model.
# 6.  **Monitoring:** Watch the model in the real world. If it starts making mistakes, go back to Step 1!
# 
# ---
# 
# # 10. COMMON CHALLENGES IN ML PROJECTS
# 
# Machine Learning is powerful, but it's not magic. Here are the "Villains" of ML:
# 
# 1.  **Bad Data (Garbage In, Garbage Out):** If your data is messy or biased, your model will be too.
# 2.  **Overfitting:** The model "memorizes" the training data but fails on new data (like a student who memorizes answers but doesn't understand the concept).
# 3.  **Underfitting:** The model is too simple and fails to see the pattern at all.
# 4.  **Data Privacy:** Ensuring user data is handled ethically and legally.
# 5.  **Compute Cost:** Large models (like ChatGPT) require expensive hardware and 
#     lots of electricity.

# # **11. QUICK SUMMARY: Regression vs Classification**
# Here is the simplest way to explain both to students in just a few lines of code.
# 

# In[53]:


# 1. REGRESSION (Predicting a Number)
from sklearn.linear_model import LinearRegression

# Data: Years of Experience -> Salary
X = np.array([[1], [2], [3]])
y = np.array([10, 20, 30])

model = LinearRegression().fit(X, y)
prediction = model.predict(np.array([[4]])) # Predict salary for 4 years

print(f"Regression Prediction for 4 years: {prediction[0]}")


# ### **Explanation of Regression Code:**
# *   **The Problem:** We want to predict a **specific number** (Salary) based on a **scale** (Experience).
# *   **The Data (`X` and `y`):** Notice `X` is a list of lists `[[1], [2], [3]]` because the model expects a 2D array (rows and columns).
# *   **`fit(X, y)`:** This is the training phase. The model calculates the mathematical relationship (the line) between experience and salary.
# *   **`predict([[4]])`:** We ask: "If someone has 4 years of experience, what will their salary be?" The result is a single continuous number.
# 

# In[54]:


# 2. CLASSIFICATION (Predicting a Category)
from sklearn.linear_model import LogisticRegression

# Data: Study Hours -> Pass(1) or Fail(0)
X = [[1], [5], [10]]
y = [0, 0, 1]

model = LogisticRegression().fit(X, y)
prediction = model.predict([[8]]) # Predict for 8 hours of study

result = "Pass" if prediction[0] == 1 else "Fail"
print(f"Classification Prediction for 8 hours: {result}")


# ### **Explanation of Classification Code:**
# *   **The Problem:** We want to predict a **category** (Pass or Fail) based on an input (Study Hours).
# *   **The Labels (`y`):** We use numbers to represent categories (`0` for Fail, `1` for Pass). This is called **Numerical Encoding**.
# *   **`LogisticRegression`:** Despite the name "Regression", this algorithm is used for finding the **probability** of belonging to a group.
# *   **The Decision:** The model doesn't give a "range" (like 75.5); it gives a binary `0` or `1`. We then translate that number back into a human word ("Pass" or "Fail").
# 
