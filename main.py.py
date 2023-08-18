import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        return Z_0[w] * (1 - np.exp(-L*X/Z_0[w]))

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        y_hat = self.get_predictons(X, Params[w], w, Params[-1])
        return np.mean((y_hat - Y)**2)
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv(data_path, sep=",")
    return df


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    # 1. Get the invalid dates
    date_regex = "^[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}$"
    invalid_dates = data[~data.Date.str.contains(date_regex)]['Date'].value_counts()
    invalid_dates

    # 2. Since Invalid dates are in format Month day-day, year we need to convert that
    from datetime import datetime
    def convert_date(curr_date):
        # parsing as 'dd/mm/yyyy' first
        try:
            return datetime.strptime(curr_date, '%d/%m/%Y').strftime('%d/%m/%Y')
        except ValueError:
            # Handle custom format 'Month day-day, year'
            splits = curr_date.split(' ')
            month = splits[0]
            days_range = splits[1].split('-')
            year = splits[2]
            
            day = days_range[0]
            
            return datetime.strptime(f"{day} {month} {year}", '%d %b %Y').strftime('%d/%m/%Y')

    # 3. Apply this to the dataframe
    data['Date'] = data['Date'].apply(convert_date)
    # data['Date'] = data['Date'].astype('datetime64[ns]')

    # Get the data of only first innings
    data = data[data['Innings'] == 1]
    
    # Consider complete matches
    comp_df = pd.DataFrame({'over_50_match' : data.groupby(data['Match'])['Over'].count() == 50, 'all_wickets': data.groupby(data['Match'])['Wickets.in.Hand'].min() == 0}).reset_index()
    comp_df['full_matches'] = comp_df['over_50_match'] | comp_df['all_wickets']
    full_matches = list(comp_df[comp_df['full_matches'] == True]['Match'])
    index_to_include = []
    for i in full_matches:
        index_to_include += list(data[data['Match'] == i].index)
    data = data.loc[index_to_include, :]
    data['Overs.Remaining'] = 50 - data['Over']

    # Required columns : 'Match',  'Wickets.in.Hand', 'Overs.Remaining', 'Runs.Remaining','Innings.Total.Runs' 
    data = pd.DataFrame(data, columns=['Match', 'Wickets.in.Hand', 'Overs.Remaining' ,'Runs.Remaining', 'Innings.Total.Runs'])
    return data


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    def find_mean_runs(df,w):
        df = df[df['Wickets.in.Hand'] == w]
        max_run = df.groupby(['Match'])['Runs.Remaining'].max()
        return np.mean(max_run)
    
    w = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

     # Initialize value for L, Z0
    for i in w:
        model.Z0[i-1] = (find_mean_runs(data, i))
    model.L = 15
    parameters = model.Z0
    parameters.append(model.L)
    
    runs = data['Runs.Remaining'].values
    overs = data['Overs.Remaining'].values
    wickets = data['Wickets.in.Hand'].values
    def minimize_function(model):
        Z0 = model[:10]
        L = model[10]
        loss = 0
        for i in range(len(wickets)):
            runs_scored = runs[i]
            remaining_overs = overs[i]
            w = wickets[i]
            if runs_scored > 0:
                predicted_run =  Z0[w - 1] * (1 - np.exp(-1*L * remaining_overs / Z0[w - 1]))
                loss = loss  + ((predicted_run - runs_scored)**2)
        return loss

        # Perform optimization
    opt = sp.optimize.minimize(minimize_function, parameters, method='L-BFGS-B')
    model.L = opt.x[-1]
    model.Z0 = opt.x[:-1]
    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """

    
    u = np.linspace(0, 50, num=300)
    predicted_values = []
    for w in range(10):
        predicted_values.append(model.get_predictions(u, model.Z0, w, model.L))

    plt.figure()
    for w in range(10):
        plt.plot(u, predicted_values[w])
    plt.title("Run Production Function Graph")
    plt.xlim((0, 50))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel('Overs Remaining')
    plt.ylabel('Average Runs Obtainable')
    plt.savefig(plot_path)
    plt.grid()
    plt.show()

    # Resource Remaining
    plt.figure()
    predict_10 = model.get_predictions(50, model.Z0, 9, model.L)
    for w in range(10):
        plt.plot(u, (predicted_values[w]/predict_10)*100)
    plt.title("Run Production Function Graph")
    plt.xlim((0, 50))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel('Overs Remaining')
    plt.ylabel('Resources Remaining')
    plt.savefig('../plots/' + 'plot_Resource_Remaining')
    plt.grid()
    plt.show()

def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    p = list(model.Z0)
    p.append(model.L)
    print(p)


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    runs = data['Runs.Remaining'].values
    overs = data['Overs.Remaining'].values
    wickets = data['Wickets.in.Hand'].values
    loss = 0
    for i in range(len(runs)):
        z = model.get_predictions(overs[i], model.Z0, wickets[i]-1, model.L)
        loss += (z - runs[i])**2
    loss /= len(runs)
    print(loss)


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
