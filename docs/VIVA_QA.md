# Viva Questions and Answers – Urban Traffic Intelligence System

**1. What is the aim of your project?**
To analyse historical urban traffic data, find patterns (peak hours, busy locations, weather effects), classify congestion and build a simple model that estimates traffic volume. It follows Data → Cleaning → EDA → Statistics → Visualization → ML → Insights.

**2. Why did you build the dashboard with Streamlit?**
It turns Python scripts into an interactive web app without HTML/JavaScript, so the focus stays on data science. It also runs locally with one command.

**3. What data cleaning did you perform and why?**
Removed duplicates (they double-count traffic), removed rows with missing or invalid traffic volume (negative values are sensor errors), converted Date/Time to proper types, and filled other missing values with the median (numbers) or most frequent value (text). Cleaning matters because wrong data gives wrong statistics.

**4. Why median instead of mean for filling missing numbers?**
The median is not affected by extreme values. Traffic data is right-skewed (a few very busy periods), so the mean would be pulled upwards.

**5. What is EDA?**
Exploratory Data Analysis: using summaries and charts to understand data before modelling – distributions, patterns, relationships and outliers.

**6. Explain mean, median and mode. Which one did you find most useful?**
Mean = average, median = middle value, mode = most frequent value. In this dataset the mean of traffic volume is higher than the median, showing right-skewed data, so the median describes a "typical" record better.

**7. What do standard deviation and variance tell you?**
They measure spread. Variance is the average squared distance from the mean; standard deviation is its square root (same units as the data). A large standard deviation means traffic changes a lot between records (here, between rush hour and night).

**8. What are quartiles and the IQR?**
Q1 and Q3 are the 25th and 75th percentiles; IQR = Q3 − Q1 is the range of the middle 50% of the data. It is resistant to outliers.

**9. What does a correlation coefficient mean?**
Pearson's r lies between −1 and +1. +1 is a perfect positive linear relationship, −1 perfect negative, 0 no linear relationship. Traffic volume and speed are negatively correlated: more vehicles, lower speed.

**10. Does correlation prove causation? Give an example from your project.**
No. Temperature correlates with traffic only because both follow the time of day (warmer in the daytime, more traffic in the daytime), not because temperature creates traffic.

**11. How did you define congestion?**
A congestion score (0–1) is built from scaled traffic volume and (inverted) scaled speed. Thresholds are mean ± 0.5 × standard deviation of the score, so they come from the data. Low/Moderate/High are therefore relative to this dataset.

**12. What is the weakness of those thresholds?**
They are relative, not absolute. A real traffic authority would use fixed limits (for example speed compared with free-flow speed).

**13. Why compare rainy and dry traffic after "adjusting"?**
Rain happens at particular hours and the volume depends on the hour. The insight divides each record by the normal volume for the same location, hour and day type, so the comparison shows the rain effect and not the rush-hour effect.

**14. Which machine learning problem is this – classification or regression?**
Regression, because the target (traffic volume) is a number. Congestion level is derived afterwards from the predicted volume.

**15. Explain Linear Regression in simple words.**
It finds the best straight-line formula, volume = intercept + (weight × feature) + ..., by minimising the squared errors. Each weight shows how much that input changes the volume.

**16. Explain Random Forest in simple words.**
It builds many decision trees on random parts of the data, each tree asks yes/no questions (for example "hour > 7?") and the forest averages all answers. Averaging reduces over-fitting and captures non-linear patterns such as rush hours.

**17. Why does Random Forest score higher than Linear Regression here?**
Traffic versus hour is not a straight line (two peaks). Linear Regression needs the hour converted to categories to cope, while the forest learns interactions like "weekday + 8 AM + city centre" automatically.

**18. What is the train/test split and why is it needed?**
The data is split (default 80/20). The model learns from the training part and is evaluated on the unseen test part. Evaluating on training data would hide over-fitting.

**19. What is over-fitting? How do you detect it?**
The model memorises the training data and performs badly on new data. Detect it by comparing training and test scores; the app warns when the training R² is more than 0.1 higher.

**20. What are R², MAE and RMSE?**
R² = share of variation explained (1 is perfect). MAE = average absolute error in vehicles. RMSE = square root of the average squared error; it punishes large mistakes more.

**21. What is data leakage? Did you avoid it?**
Leakage is using information in training that would not be available at prediction time. Average speed and accidents were not used as inputs because they are only known after the traffic occurs.

**22. Why one-hot encoding?**
Models need numbers. One-hot encoding turns a category such as Location into one 0/1 column per value, avoiding a false order (e.g. "Highway" > "University").

**23. Why scale the numeric features?**
StandardScaler puts features on a similar scale. It matters for Linear Regression and is harmless for Random Forest.

**24. Can your predictions be trusted?**
They are estimates based on historical data only. They are useful for learning and rough planning but not guaranteed, and the sample data is synthetic. A real project would need real data, time-based validation and more features.

**25. How does the app handle missing columns?**
Only Traffic_Volume is required. Each page checks which columns exist and shows a message instead of an error when a chart cannot be drawn.

**26. How would you improve the project?**
Use real city data, split train/test by date, add holidays/events, try a classifier for congestion levels, add cross-validation and hyper-parameter tuning.

**27. Why is the sample dataset not "just random"?**
It is generated from rules: rush-hour peaks, weekend reduction, location capacity, speed that falls near capacity and rain that lowers volume, plus random noise. That makes the patterns testable.
