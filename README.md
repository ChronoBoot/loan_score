# Loan Prediction Project

This project is centered around using a random forest model to predict loan approvals. It involves downloading and processing data from Azure Blob Storage, and leveraging the power of pandas and machine learning for prediction analysis.

## Getting Started

These instructions will guide you through setting up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher

#### Installing Python
If you don't have Python installed, download and install it from the [Python official website](https://www.python.org/downloads/). Ensure you select the version that corresponds to your operating system. During installation, remember to check the option that says 'Add Python to PATH' to make Python accessible from the command line.

### Installing

Follow these steps to get your development environment up and running:

1. Clone the repository:
    ```bash
    git clone https://github.com/ChronoBoot/loan_score.git
    ```
2. Navigate to the project directory:
    ```bash
    cd loan_score
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On MacOS/Linux:
     ```bash
     source venv/bin/activate
     ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

Note: To exit the virtual environment, simply type `deactivate` in the command line.

## Usage

To run the project:
1. Ensure the virtual environment is activated.
2. Execute the backend main script in the root folder:
    ```bash
    python -m backend.src.main
    ```
3. Open a new terminal and activate the virtual environment:
    - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On MacOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Execute the frontend main script in the root folder:
    ```bash
    python -m frontend.src.main
    ```

This will start the process of data retrieval, cleaning, and prediction. The output will be your predicted loan approvals.

## Contributing

If you wish to contribute to this project, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT license - see the LICENSE file for details.

## Acknowledgments

- Thanks to Kaggle(https://www.kaggle.com/competitions/home-credit-default-risk/overview) for providing the dataset.
- A special thanks to Morgan Gautherot (https://www.linkedin.com/in/morgan-gautherot-phd-05a227108/) for his great help on this project.

## Model Development Process

In this phase of the project, we implemented a simple Random Forest model. The primary dataset used was the `application_train` table, focusing on encoding non-numerical values to fit the model requirements.
Data is aggregated from the other tables grouping by loan and merge with the data from `application_train`.
This approach yielded a promising precision rate of around 92%, marking a solid start for our predictive modeling. The aim was to establish a baseline performance, using a well-known and reliable machine learning algorithm. This initial step serves as a foundation for more complex model development and optimization in the future.