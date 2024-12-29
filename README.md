# Frequent Itemsets and Association Rules

This project analyzes electric vehicle data to identify frequent itemsets and generate association rules using the Apriori algorithm. Below are the steps for running the code and understanding the results.

## Prerequisites

Ensure you have the following installed in your Python environment:

- `pandas`
- `mlxtend`

You can install the required packages using the following command:

```bash
pip install pandas mlxtend
```

## Dataset

The analysis is based on the dataset `Updated_Electric_Vehicle_Data_VIN.csv`. Ensure the file is present in the same directory as the script.

### Columns Used

- **County**
- **City**
- **State**
- **Model Year**
- **Make**
- **Model**
- **Electric Vehicle Type**
- **Clean Alternative Fuel Vehicle CAFV Eligibility**

## Steps Performed

1. **Data Preprocessing**:
   - Missing values are removed.
   - Selected columns are combined into transactions (list format).

2. **Transaction Encoding**:
   - Transactions are transformed into a binary matrix using `TransactionEncoder`.

3. **Frequent Itemsets**:
   - The Apriori algorithm is applied with a minimum support of `0.2`.
   - Results are saved to `frequent_itemsets.csv` if frequent itemsets are found.

4. **Association Rules**:
   - Rules are generated from the frequent itemsets with a minimum confidence of `0.7`.
   - Results are sorted by confidence and saved to `association_rules.csv`.

## Running the Code

1. Place the `Updated_Electric_Vehicle_Data_VIN.csv` dataset in the project directory.
2. Execute the script using Python:

   ```bash
   python script_name.py
   ```

3. Check the terminal output for frequent itemsets and top association rules.
4. Output files (`frequent_itemsets.csv` and `association_rules.csv`) will be generated in the project directory.

## Output Explanation

### Frequent Itemsets

- Shows combinations of items that appear together frequently.
- Example:

  | Itemset        | Support |
  |----------------|---------|
  | {'Model A'}    | 0.25    |

### Association Rules

- Provides relationships between items, including metrics such as support, confidence, and lift.
- Example:

  | Antecedents | Consequents | Support | Confidence | Lift |
  |-------------|-------------|---------|------------|------|
  | {'Model A'} | {'City X'}  | 0.25    | 0.8        | 1.5  |

## Customization

You can adjust the following parameters:

- **Minimum Support**: Change the value of `min_support` in the Apriori function.
- **Minimum Confidence**: Modify `min_threshold` in the `association_rules` function.

## Notes

- Ensure the dataset has no missing or inconsistent data before running the script.
- The output files can be used for further analysis or visualization.

## Contact

For issues or suggestions, please contact [Your Name/Email].

