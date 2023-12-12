# Linear Regression Tool (LRT)

## Overview

The Linear Regression Tool (LRT) is a command-line Python script designed for performing linear regression analysis on datasets. It provides functionality for training linear regression models using both Batch Gradient Descent (bgd) and Stochastic Gradient Descent (sgd) algorithms.

## Features

- **Batch Gradient Descent and Stochastic Gradient Descent:** Choose between two gradient descent algorithms.
- **Customizable Learning Parameters:** Adjust learning rate, the number of iterations, and convergence threshold.
- **Interactive Mode:** Execute the linear regression for a given feature interactively.

## Prerequisites

Make sure you have Python installed. If not, you can download it from [python.org](https://www.python.org/downloads/).

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/codeonym/LRT.git
    ```

2. Navigate to the project folder:

    ```bash
    cd LRT
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

   ```bash
   python linearRegressionTool.py datafile [options]
   ```
1. Example

   ```bash
   python linearRegressionTool.py dataset.csv -a sgd -l 0.01 -i 1000 -c 1e-5 -p save -e
   ```
2. Options

- **datafile:** Path to the dataset file.
- **-a, --algorithm:** Learning algorithm (choices: bgd, sgd; default: sgd).
- **-l, --learning-rate:** Learning rate for gradient descent (default: 0.01).
- **-i, --iterations:** Number of iterations for gradient descent (default: 1000).
- **-c, --convergence:** Convergence threshold for the cost function (default: 1e-5).
- **-p, --plot:** Plot the linear regression line (choices: save, show; default: None).
- **-e, --execute:** Execute the linear regression for a given feature.

### Acknowledgments

- [Colorama](https://pypi.org/project/colorama/) - for colored terminal output.
- [NumPy](https://numpy.org/) - for numerical computations.
- [Pandas](https://pandas.pydata.org/) - for data manipulation.
- [Matplotlib](https://matplotlib.org/) - for data visualization.
