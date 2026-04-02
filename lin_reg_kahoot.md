# 🍋 Linear Regression Kahoot Quiz - HERE AND NOW AI

## "AI is Good"

Get ready to test your knowledge about Linear Regression! All questions are based on our Lemonade Stand story and the logic from `day3_linear_regression_plus_new_plan.ipynb`.

---

### Phase 1: The Basics (The Lemonade Stand Story)

1. **In our story, what does the "X" usually represent?**
   - A) Total Profit
   - B) Number of Lemons (The Feature) ✅
   - C) Sidewalk Money
   - D) The Robot's name

2. **In our story, what does the "y" represent?**
   - A) Number of Lemons
   - B) The Color of the Stand
   - C) Total Money made (The Target) ✅
   - D) The "Save Point"

3. **What is the Greek letter "θ" (Theta) used for in Linear Regression?**
   - A) The Temperature
   - B) The Secret Rules/Numbers the robot learns ✅
   - C) The number of customers
   - D) The exit button

4. **In the formula y = 4 + 3x, what is the "Bias" (θ₀)?**
   - A) 3
   - B) x
   - C) 4 ✅
   - D) y

5. **In the formula y = 4 + 3x, what is the "Weight" (θ₁)?**
   - A) 4
   - B) 3 ✅
   - C) x
   - D) The Weight of the lemons

---

### Phase 2: Building the Data

6. **Why do we use `np.random.seed(42)`?**
   - A) To grow real lemons
   - B) To make our "random" numbers reproducible ✅
   - C) To make the code run slower
   - D) To delete all files

7. **What does `np.random.rand(100, 1)` create?**
   - A) 100 random colors
   - B) A single number 100
   - C) 100 random numbers between 0 and 1 ✅
   - D) 1 random number 100 times

8. **What happens when we calculate `2 * np.random.rand(100, 1)`?**
   - A) We multiply 100 by 2
   - B) We stretch the range to 0 and 2 lemons ✅
   - C) We get only the number 2
   - D) It causes an error

9. **What is the "Noise" (`np.random.randn`) in our data?**
   - A) Loud music at the stand
   - B) Fake data points
   - C) The "Wobble" or real-life messiness ✅
   - D) A counting error

10. **A "perfect" straight line in real data is...**
    - A) Always expected
    - B) Very rare (real data has noise) ✅
    - C) Impossible to draw
    - D) Only for circles

---

### Phase 3: The Magic Formula (Normal Equation)

11. **What is the "Normal Equation" used for?**
    - A) To reset the computer
    - B) To find the best rules instantly using math ✅
    - C) To count the lemons
    - D) To draw pretty pictures

12. **The "θ best" in our code represents...**
    - A) The worst possible guess
    - B) The mathematical "Gold Medal" guess ✅
    - C) A random number
    - D) The color of the chart

13. **Why do we `add_dummy_feature` (adding a 1 to every row)?**
    - A) To make the list longer
    - B) To help the robot find the "Sidewalk Money" (Bias) ✅
    - C) Because 1 is a lucky number
    - D) To confuse the robot

14. **What does `np.linalg.inv` do in simple terms?**
    - A) It invents a new number
    - B) It works "backwards" to solve the puzzle ✅
    - C) It makes the numbers invisible
    - D) It deletes the data

15. **The "@" symbol in `X_b.T @ y` stands for...**
    - A) Email address
    - B) At the stand
    - C) Matrix Multiplication ✅
    - D) Addition

---

### Phase 4: Training and Prediction

16. **What is a "Prediction" in Linear Regression?**
    - A) Looking at the past
    - B) Using a rule to guess the future ✅
    - C) Playing a game
    - D) Changing the data

17. **If the robot learns y = 5 + 2x, what is its guess for x = 10?**
    - A) 15
    - B) 20
    - C) 25 ✅
    - D) 50

18. **What does `LinearRegression()` from Scikit-Learn do?**
    - A) Cleans the lemons
    - B) Automates the learning process ✅
    - C) Opens a browser
    - D) Shuts down the computer

19. **What is the "Intercept" in Scikit-Learn?**
    - A) θ₁ (The Slope)
    - B) The Noise
    - C) θ₀ (The Bias / Sidewalk Money) ✅
    - D) The Number of Lemons

20. **What are "Coefficients" (`coef_`) in Scikit-Learn?**
    - A) The Sidewalk Money
    - B) The Weights / Slope of our line ✅
    - C) The Error score
    - D) The number of days

---

### Phase 5: Gradient Descent (The Treacherous Mountain)

21. **What is "Gradient Descent" analogous to?**
    - A) Climbing a tree
    - B) Finding the bottom of a foggy valley in steps ✅
    - C) Racing a car
    - D) Jumping off a cliff

22. **What is the "Learning Rate" (η / Eta)?**
    - A) How fast the lemons grow
    - B) The size of the robot's steps towards the bottom ✅
    - C) The price of a lemon
    - D) The volume of the computer

23. **What happens if the Learning Rate is TOO LARGE?**
    - A) The robot learns too well
    - B) The robot "jumps over" the bottom and gets lost ✅
    - C) The code stops
    - D) Nothing happens

24. **What happens if the Learning Rate is TOO SMALL?**
    - A) Training takes a very long time ✅
    - B) The robot explodes
    - C) The line gets too steep
    - D) The data disappears

25. **What is "Batch" Gradient Descent?**
    - A) Looking at one lemon at a time
    - B) Looking at ALL the data before taking one step ✅
    - C) Coding in batches
    - D) Baking a batch of cookies

---

### Phase 6: Stochastic and Mini-Batch

26. **What does "Stochastic" mean in Gradient Descent?**
    - A) High Quality
    - B) Using a Random single dot to take a step ✅
    - C) Very slow
    - D) A type of lemon

27. **Why is Stochastic GD "Jittery" or jumpy?**
    - A) Because the computer is cold
    - B) Because it reacts to every single random dot ✅
    - C) Because there is too much data
    - D) It isn't jittery

28. **What is "Mini-Batch" Gradient Descent?**
    - A) A small lemonade stand
    - B) Learning from small groups (clusters) of dots ✅
    - C) Learning from nothing
    - D) A very short robot

29. **Which method is usually the "Best of Both Worlds"?**
    - A) Batch GD
    - B) Stochastic GD
    - C) Mini-Batch GD ✅
    - D) Drawing by hand

30. **What is the ultimate goal of Linear Regression?**
    - A) To draw a circle
    - B) To minimize the Error (MSE) and find the best line ✅
    - C) To make 1000 dollars
    - D) To make the charts blue

---

### 🤝 Connect with HERE AND NOW AI
- **Website**: [hereandnowai.com](https://hereandnowai.com)
- **LinkedIn**: [HERE AND NOW AI](https://www.linkedin.com/company/hereandnowai/)
- **Instagram**: [@hereandnow_ai](https://instagram.com/hereandnow_ai)
- **YouTube**: [HERE AND NOW AI](https://youtube.com/@hereandnow_ai)
- **Email**: [info@hereandnowai.com](mailto:info@hereandnowai.com)
- **Slogan**: *AI is Good*
