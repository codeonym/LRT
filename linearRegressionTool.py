# IMPORTS
import time
from argparse import ArgumentParser
import numpy as np
from sys import stderr, exit
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
from colorama import Fore, Style, init


# FUNCTIONS
# CSV VALIDATOR
def is_valid_csv(file_path):
    # TRIES TO READ THE CSV FILE
    error = ''
    try:
        with open(file_path, "r") as file:
            csv.reader(file).__next__()
            # SUCCESS
        return True
    except csv.Error:
        error = f"{file_path} is not a valid CSV file\n"
    except FileNotFoundError:
        error = f"{file_path} not found\n"
    except PermissionError:
        error = f"{file_path} permission denied\n"

    # FAIL
    error_message = f"Error: {error}"
    stderr.write(f"{Fore.RED}{Style.BRIGHT}{error_message}{Style.RESET_ALL}")
    return False


# LOADING DATASET FROM FILE
def load_dataset(file_path):
    try:

        # READING FROM CSV WITH PANDAS
        dataset = pd.read_csv(file_path)

        # GETTING THE FEATURES VARIABLE
        X = dataset.iloc[:, : -1].values

        # GETTING THE TARGET VARIABLE
        y = dataset.iloc[:, -1].values

        # GETTING COLUMN NAMES
        column_names = dataset.columns.tolist()

        # CREATE LABELS DICTIONARY
        labels = {'x': column_names[:-1], 'y': column_names[-1]}
        # SUCCESS
        return X, y, labels

    except Exception as e:
        # ERROR OCCURRED
        error_message = f"Error: Something Went Wrong While Reading Dataset from {file_path}\n"
        stderr.write(f"{Fore.RED}{Style.BRIGHT}{error_message}{Style.RESET_ALL}")

        # EXITING THE PROGRAM
        exit(-1)


# COST FUNCTION
def compute_cost(X, y, theta):

    # CALCULATING HYPOTHESIS
    h_theta = np.dot(X, theta)

    # CALCULATES THE COST Ho(Xi) - Yi for i in range(1, m)
    return (1 / 2) * np.sum((h_theta - y) ** 2)


# HYPOTHESIS FUNCTION
def h_theta_func(X, theta):

    # CALCULATE ho(x) = sum of Xi.THETA_i in range (0, n)
    return np.dot(X, theta)


# BATCH GRADIENT DESCENT ALGORITHM
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000, convergence_threshold=1e-5):

    # RETURNS [ n_rows, n_cols ]
    m, n = np.shape(X)

    # CREATE AN ARRAY FILLED WITH 0s
    theta = np.zeros(n)

    # ASSIGN INFINITY TO THE PREVIOUS COST
    prev_cost = float("inf")

    iterates = 0
    # LOOPING UNTIL EITHER REACHING ~CONVERGENCE~ OR EXCEEDING ~ITERATION'S COUNT~
    for _ in range(iterations):

        iterates = _

        # CALCULATING THE HYPOTHESIS
        h_theta = h_theta_func(X, theta)

        # UPDATING PARAMETERS USING THE GRADIENT DESCENT
        for j in range(n):

            # FOR EACH PARAMETER (THETA_j):
            # THETA_j := THETA_j - ALPHA * (h_theta(Xi) - Yi) * Xj  -- for i in range (1, m)
            gradient_j = np.sum((h_theta - y) * X[:, j])
            theta[j] -= learning_rate * gradient_j

        # CALCULATING THE NEW COST
        cost = compute_cost(X, y, theta)

        # CHECK FOR CONVERGENCE
        if (prev_cost - cost) < convergence_threshold:

            # CONVERGED
            break

        # UPDATING THE PREVIOUS COST
        prev_cost = cost
    return theta, iterates


# STOCHASTIC GRADIENT DESCENT ALGORITHM
def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=1000, convergence_threshold=1e-5):

    # RETURNS [ n_rows, n_cols ]
    m, n = np.shape(X)

    # CREATE AN ARRAY FILLED WITH 0s
    theta = np.zeros(n)

    # ASSIGN INFINITY TO THE PREVIOUS COST
    prev_cost = float("inf")

    iterates = 0
    # LOOPING UNTIL EITHER REACHING ~CONVERGENCE~ OR EXCEEDING ~ITERATION'S COUNT~
    for _ in range(iterations):

        iterates = _
        for i in range(m):

            # CALCULATING THE HYPOTHESIS h_theta(Xi): FOR i-th X ONLY
            h_theta_i = h_theta_func(X[i, :], theta)

            # UPDATING PARAMETERS USING THE GRADIENT DESCENT
            for j in range(n):

                # FOR EACH PARAMETER (THETA_j):
                # THETA_j := THETA_j - ALPHA * (h_theta(Xi) - Yi) * Xi,j
                gradient_j = (h_theta_i - y[i]) * X[i, j]
                theta[j] -= learning_rate * gradient_j

        # CALCULATING THE NEW COST
        cost = compute_cost(X, y, theta)

        # CHECK FOR CONVERGENCE
        if (prev_cost - cost) < convergence_threshold:

            # CONVERGED
            break

        # UPDATING THE PREVIOUS COST
        prev_cost = cost
    return theta, iterates


# PLOTTING DATA
def plot_data_2d(X, y, theta, file, labels=None):

    plt.scatter(X[:, 0], y, label="Data Points")

    # PLOTTING REGRESSION LINE
    regression_line = h_theta_func(X[:, 0], theta[0])
    plt.plot(X[:, 0], regression_line, color='red', label='Regression Line')

    # LABELS NAMING
    if labels is None:
        plt.xlabel("Feature")
        plt.ylabel("Target")
    else:
        plt.xlabel(labels['x'][0])
        plt.ylabel(labels['y'])

    plt.legend()
    plt.title(f'Plot Of Dataset {file}')

    return plt


def plot_data_3d(X, y, theta, file, labels=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, label="Data Points")

    x_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_values = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = theta[0] * x_mesh + theta[1] * y_mesh

    ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, color='red', label='Regression Plane')

    # LABELS NAMING
    if labels is None:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target")
    else:
        ax.set_xlabel(labels["x"][0])
        ax.set_ylabel(labels["x"][1])
        ax.set_zlabel(labels["y"])
    ax.legend()
    ax.set_title(f'3D Plot Of Dataset {file}')

    return plt


# PLOTTING HANDLER
def plot_data(X, y, theta, filename, labels, dim, action="save"):

    plot = None
    # CHECK DIMENSION FOR PLOTTING
    match dim:
        case 1:
            # VALID 2D DIMENSION
            plot = plot_data_2d(X, y, theta, filename, labels)
        case 2:
            # VALID 3D DIMENSION
            plot = plot_data_3d(X, y, theta, filename, labels)
        case _:
            # CURRENTLY UNAVAILABLE
            error_message = f"Error: cannot plot {dim}-dimensions diagram \n"
            stderr.write(f"{Fore.RED}{Style.BRIGHT}{error_message}{Style.RESET_ALL}")

    # ACTION HANDLING
    if plot is not None:
        match action:
            case "save":
                # VISUALIZING DATA INTO OUTPUT FOLDER
                basename = os.path.splitext(filename)[0] + '.png'
                plot.savefig(f'output/{basename}')
            case "show":
                # VISUALIZING DATA INTO OUTPUT STREAM
                plot.show()


# STATISTICS
def print_statistics(y, predictions, iterations, theta, exec_time):
    # Calculate metrics
    r2 = calculate_r_squared(y, predictions)

    # Print a fancy header
    print("\n" + "=" * 40)
    print(f"{Fore.YELLOW}           STATISTICS SUMMARY{Style.RESET_ALL}")
    print("=" * 40 + "\n")

    # Display statistics with colors
    print(f"{Fore.BLUE}Total Iterations:{Style.RESET_ALL} {iterations}")
    print(f"{Fore.BLUE}Theta:{Style.RESET_ALL} {theta}")
    print(f"{Fore.BLUE}Total Execution Time:{Style.RESET_ALL} {exec_time:.4f}s")
    print(f"{Fore.BLUE}R-squared (R2) Score:{Style.RESET_ALL} {r2:.4f}")

    # Print a fancy footer
    print("\n" + "=" * 40 + "\n")


def calculate_r_squared(y, predictions):

    mean_y = sum(y) / len(y)
    ss_total = sum((y - mean_y)**2)
    ss_residual = sum((y - predictions)**2)
    return 1 - (ss_residual / ss_total)


# TEST USER CUSTOM INPUT FEATURE
def test_algo(theta, labels, n):
    user_ifeature = []

    # Prompt user for input features
    print(f"{Fore.YELLOW}Input Your Features:{Style.RESET_ALL}")
    for i in range(n):
        feature_name = labels["x"][i]
        feature_value = float(input(f'{Fore.CYAN}{feature_name}:{Style.RESET_ALL} '))

        user_ifeature.append(feature_value)

    # Display a loading message
    print(f'\n{Fore.BLUE}Calculating...{Style.RESET_ALL}')

    # Make predictions for user input features
    user_predictions = h_theta_func(user_ifeature, theta)

    # Display the result
    print(f'\n{Fore.GREEN}Your Generated Target:{Style.RESET_ALL} {user_predictions}\n')


# MAIN FUNCTION
def main():
    init(autoreset=True)
    # INIT ARGUMENT-PARSER
    parser = ArgumentParser(prog="LRT-1.0", description="Linear Regression Tool")

    # SETTING ARGUMENTS
    parser.add_argument("datafile",
                        help="Path to the dataset file")
    parser.add_argument("-a", "--algorithm", choices=["bgd", "sgd"], default="sgd",
                        help="Learning algorithm: batch gradient descent (bgd) or stochastic gradient descent (sgd)")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.01,
                        help="Learning rate for gradient descent")
    parser.add_argument("-i", "--iterations", type=int, default=1000,
                        help="Number of iterations for gradient descent")
    parser.add_argument("-c", "--convergence", type=float, default=1e-5,
                        help="Convergence threshold for the cost function")
    parser.add_argument("-p", "--plot", choices=["save", "show"],nargs="?", const="save", default=None,
                        help="Plot the linear regression line: save to output folder or show in output stream")
    parser.add_argument("-e", "--execute", "-exec", action="store_true",
                        help="Execute the linear regression for a given feature")

    # PARSING INTO NAMESPACE ARGS
    args = parser.parse_args()

    # VALIDATION
    if not is_valid_csv(args.datafile):
        exit(-1)

    # LOADING DATASET FROM FILE
    X, y, labels = load_dataset(args.datafile)

    # GETTING THE DIMENSIONS
    m, n = np.shape(X)
    # INIT THETA AS ARRAY OF ZEROS
    theta = np.zeros(n)
    # GETTING THE CSV FILE NAME
    filename = os.path.basename(args.datafile)

    # PROCEEDING
    # CALCULATING EXECUTION TIME
    start_time = time.time()
    iterations = 0
    if args.algorithm == "bgd":
        # -- TRAINING WITH BGD
        theta, iterations = batch_gradient_descent(X, y, args.learning_rate, args.iterations, args.convergence)
    else:
        # -- TRAINING WITH SGD
        theta, iterations = stochastic_gradient_descent(X, y, args.learning_rate, args.iterations, args.convergence)

    end_time = time.time()

    # EXECUTION TIME :
    execution_time = end_time - start_time

    # Use the trained model to make predictions
    predictions = h_theta_func(X, theta)

    # Visualize statistics
    print_statistics(y, predictions, iterations, theta, execution_time)

    # EXECUTE CUSTOM FEATURE
    if args.execute:
        test_algo(theta, labels, n)

    if args.plot:

        # PLOTTING THE DATASET
        plot_data(X, y, theta, filename, labels, n, args.plot)

    return


if __name__ == "__main__":
    main()
