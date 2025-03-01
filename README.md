# Mean Reversion and Copula-Based Trading Strategy

This repository implements a **mean reversion trading strategy** using **statistical arbitrage** and **copula-based dependence modeling** to identify profitable trading opportunities in cryptocurrency pairs.

## **1. Overview**
The strategy consists of two main components:
1. **Mean Reversion Trading**: Identifies pairs that exhibit mean-reverting behavior using **rolling OLS regression** and **cointegration tests**.
2. **Copula-Based Modeling**: Detects non-linear dependencies between asset pairs and uses **conditional probability** to enhance trade decisions.

## **2. Data Preprocessing**
- Loads cryptocurrency price data from CSV files.
- Merges the closing prices into a single dataframe.
- Applies **rank transformation** (empirical CDF) to normalize data.
- Splits data into **training (60%)** and **testing (40%)** sets.

## **3. Mean Reversion Trading Strategy**
- **Pairs Selection**: Identifies asset pairs with **cointegration tests**.
- **Spread Calculation**: Uses **Rolling OLS Regression** to compute spreads.
- **Trading Rules**:
  - **Short the spread** when it is above the upper threshold.
  - **Long the spread** when it is below the lower threshold.
  - **Exit** when the spread returns to the mean.
- **Performance Evaluation**:
  - Selects the best time window based on total **PnL**.
  - Ranks the top 20 pairs based on **profitability and cointegration strength**.
  - Evaluates **trade success rate** on the testing set.

## **4. Copula-Based Dependence Modeling**
- Fits five different **copula models**:
  - **Gaussian, t-Copula, Clayton, Gumbel, Frank**
- Selects the best copula based on **Akaike Information Criterion (AIC)**.
- Uses **conditional probability** to refine trade signals:
  - If the **probability of mean reversion is high**, enter a **trade**.
  - If the **probability is low**, **exit** the trade.

## **5. Expected Results and Key Outputs**
- Identifies **top 5 copula-fitted pairs**.
- Generates **trade signals** using **copula-implied probabilities**.
- Computes **overall strategy success rate**.


## **6. The Code sequence of the Strategy**

The following outlines the step-by-step process of the code execution:

### **a. Read Data Files**
- Identify all `.csv` files in the directory.
- Extract instrument names from filenames.
- Read the `close` price column and store it in a dictionary.

### **b. Data Preprocessing**
- Merge all individual instrument data into a single DataFrame.
- Drop rows with missing values.
- Convert price data into cumulative distribution function (CDF) values.

### **c. Train-Test Split**
- Split the data into 60% training and 40% testing.

### **d. Mean Reversion Strategy Implementation**
- Compute rolling mean and standard deviation of the spread.
- Define entry and exit conditions for trades.
- Execute trades based on threshold conditions.
- Calculate cumulative PnL (profit and loss).

### **e. Optimal Window Selection**
- Iterate over trading pairs to determine the best rolling window.
- Perform rolling OLS regression to model the spread.
- Simulate mean reversion strategy for each window.
- Select the best window based on total PnL.

### **f. Top 20 Trading Pairs Selection**
- Rank pairs based on profitability.
- Compute cointegration p-values for the top 20 pairs.

### **g. Mean Reversion Trade Success Rate**
- Apply the strategy to the test dataset.
- Compute the success rate for each pair.

### **h. Copula-Based Dependence Modeling**
- Fit multiple copula models (Gaussian, t-Copula, Clayton, Gumbel, Frank).
- Select the best copula model for each pair based on AIC (Akaike Information Criterion).
- Identify the top 5 best-fitted copula pairs.

### **i. Trading Strategy Based on Conditional Probabilities**
- Use copula models to estimate conditional probabilities.
- Generate trading signals based on probability thresholds.

### **j. Output Results**
- Print the top 20 trading pairs with cointegration p-values.
- Print mean reversion success rates.
- Display the top 5 copula-fitted pairs.
- Generate trading signals for copula-based strategies.


## **7. Results and Key Findings**

### **7.1 Mean Reversion Trade Success Rates**
The strategy was applied to multiple cryptocurrency pairs, and the **top 20 performing pairs** are listed below. The success rate represents the percentage of profitable trades executed during the backtesting period.

| Pair         | Best Window | PnL   | Success Rate |
|-------------|------------|-------|--------------|
| ADA-BCH     | 1 week     | 100.00 | 94.87%       |
| ADA-BTC     | 1 week     | 100.00 | 97.87%       |
| ADA-FLM     | 1 week     | 100.00 | 97.87%       |
| ADA-GALA    | 1 week     | 100.00 | 87.50%       |
| AGLD-ALGO   | 1 week     | 100.00 | 97.14%       |
| AGLD-FLOKI  | 1 week     | 100.00 | 91.30%       |
| ALGO-FLM    | 1 week     | 100.00 | 98.15%       |
| ALGO-FLOKI  | 1 week     | 100.00 | 86.21%       |
| APE-BONK    | 1 week     | 100.00 | 85.00%       |
| APE-GFT     | 1 week     | 100.00 | 92.11%       |
| BADGER-DGB  | 1 week     | 100.00 | 96.23%       |
| BADGER-GALA | 1 week     | 100.00 | 97.87%       |
| BAL-BAND    | 1 week     | 100.00 | 97.30%       |
| BAL-CHZ     | 1 week     | 100.00 | 97.22%       |
| BAL-DOGE    | 1 week     | 100.00 | 92.59%       |
| BAL-FLM     | 1 week     | 100.00 | 100.00%      |
| BAL-GFT     | 1 week     | 100.00 | 96.97%       |
| BAND-GFT    | 1 week     | 100.00 | 94.12%       |
| BAT-FLOKI   | 1 week     | 100.00 | 93.33%       |
| CSPR-GFT    | 1 week     | 100.00 | 93.10%       |

The **average success rate across the top 20 pairs** was **94.34%**, indicating that the mean reversion strategy performed well in capturing profitable trading opportunities.

### **7.2 Copula-Based Strategy Limitations**
While the **copula method** was tested for trade signal refinement, it **ran into RAM issues** due to the computational complexity of high-dimensional probability distributions. Future optimizations should focus on:
- **Reducing data dimensionality** before applying copula models.
- **Parallelizing computations** to handle large datasets.
- **Experimenting with alternative dependence structures** that require less memory.

### **7.3 Key Takeaways**
- **Mean reversion trading with cointegration-based pairs** showed strong profitability.
- **The 1-week rolling window** was consistently the best across pairs.
- **Success rates above 90%** were observed for most pairs, validating the robustness of the strategy.
- **Copula-based modeling requires further optimization** to be practical for real-time applications.

