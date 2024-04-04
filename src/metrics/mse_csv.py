import pandas as pd
import matplotlib.pyplot as plt

def calculate_errors(dataframe, column_a, column_b):
    # Calculate Mean Squared Error (MSE)
    mse = ((dataframe[column_a] - dataframe[column_b]) ** 2).mean()

    # Calculate Mean Absolute Error (MAE)
    mae = (dataframe[column_a] - dataframe[column_b]).abs().mean()

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = ((dataframe[column_a] - dataframe[column_b]) / dataframe[column_a]).abs().mean() * 100

    return mse, mae, mape

def main():
    # Read the CSV file into a DataFrame
    csv_file = "input/carbon_specs_all_use.csv"
    # csv_file = "input/use_hp_combined.csv"

    df = pd.read_csv(csv_file)

    # df = df[df['Use Carbon Footprint'] <= 2000]

    # Specify the column names
    column_a = 'Use Carbon Footprint'  # Ground truth
    column_b = 'Fit Estimated Use Carbon Footprint'  # Predicted value

    # Calculate errors
    mse, mae, mape = calculate_errors(df, column_a, column_b)

    # Print the results
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape, "%")

    # Plot actual vs. predicted values
    plt.scatter(df[column_a], df[column_b], alpha=0.5)
    plt.plot([df[column_a].min(), df[column_a].max()], [df[column_a].min(), df[column_a].max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.savefig('output/fit_use_carbon.png')  # Adjust the path as needed
    # plt.show()

if __name__ == "__main__":
    main()
