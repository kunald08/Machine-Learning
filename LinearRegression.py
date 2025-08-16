import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

# -----------Linear Regression-----------
class LinearRegression: 
    """
    Linear Regression implementation from scratch using gradient descent.
    Implements the model y = mx + c (slope-intercept form).
    """
    def __init__(self): 
        """
        Initialize the model with empty parameters dictionary.
        Parameters will contain 'm' (slope) and 'c' (intercept).
        """
        self.parameters = {} 

    def forward_propagation(self, train_input): 
        """
        Forward pass to make predictions using current parameters.
        Applies the linear equation: y = mx + c
        
        Args:
            train_input: Input features (x values)
            
        Returns:
            predictions: Predicted outputs (y values)
        """
        m = self.parameters['m'] 
        c = self.parameters['c'] 
        predictions = np.multiply(m, train_input) + c 
        return predictions 

    def cost_function(self, predictions, train_output): 
        """
        Calculate the Mean Squared Error (MSE) cost.
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            predictions: Model predictions
            train_output: Actual target values
            
        Returns:
            cost: Mean squared error
        """
        cost = np.mean((train_output - predictions) ** 2) 
        return cost 

    def backward_propagation(self, train_input, train_output, predictions): 
        """
        Calculate gradients of the cost function with respect to parameters.
        Implements the partial derivatives of MSE with respect to m and c.
        
        Args:
            train_input: Input features (x values)
            train_output: Actual target values
            predictions: Model predictions
            
        Returns:
            derivatives: Dictionary containing gradients for m and c
        """
        derivatives = {} 
        df = (predictions-train_output) 
        # Gradient of cost w.r.t. m: 2/n * Σ(x * (y_pred - y_true))
        dm = 2 * np.mean(np.multiply(train_input, df)) 
        # Gradient of cost w.r.t. c: 2/n * Σ(y_pred - y_true)
        dc = 2 * np.mean(df) 
        derivatives['dm'] = dm 
        derivatives['dc'] = dc 
        return derivatives 

    def update_parameters(self, derivatives, learning_rate): 
        """
        Update model parameters using gradients and learning rate.
        Implements gradient descent update rule: param = param - learning_rate * gradient
        
        Args:
            derivatives: Dictionary containing gradients for m and c
            learning_rate: Step size for gradient descent
        """
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm'] 
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self, train_input, train_output, learning_rate, iters): 
        """
        Train the linear regression model using gradient descent.
        Includes visualization of the training process as an animation.
        
        Args:
            train_input: Input features (x values)
            train_output: Actual target values
            learning_rate: Step size for gradient descent
            iters: Number of iterations for training
            
        Returns:
            parameters: Final model parameters (m and c)
            loss: List of cost values during training
        """
        # Initialize parameters with random negative values
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

        self.loss = [] 

        # Set up the visualization
        fig, ax = plt.subplots() 
        x_vals = np.linspace(min(train_input), max(train_input), 100) 
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'], color='red', label='Regression Line') 
        ax.scatter(train_input, train_output, marker='o', color='green', label='Training Data') 

        ax.set_ylim(0, max(train_output) + 1) 

        def update(frame): 
            """
            Update function for animation.
            Performs one iteration of the training process.
            
            Args:
                frame: Animation frame number
                
            Returns:
                line: Updated line object for animation
            """
            # Forward pass
            predictions = self.forward_propagation(train_input) 
            # Calculate current loss
            cost = self.cost_function(predictions, train_output) 
            # Calculate gradients
            derivatives = self.backward_propagation(train_input, train_output, predictions) 
            # Update parameters using gradient descent
            self.update_parameters(derivatives, learning_rate) 
            # Update the regression line visualization
            line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c']) 
            # Store loss for tracking progress
            self.loss.append(cost) 
            print("Iteration = {}, Loss = {}".format(frame + 1, cost)) 
            return line, 

        # Create animation for visualizing training progress
        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True) 
        ani.save('linear_regression_A.gif', writer='ffmpeg') 

        # Configure plot labels and display
        plt.xlabel('Input') 
        plt.ylabel('Output') 
        plt.title('Linear Regression') 
        plt.legend() 
        plt.show() 

        return self.parameters, self.loss



# -------------------- Data Loading and Model Training --------------------

# Load the dataset from CSV file
data = pd.read_csv('data_for_lr.csv')

# Preprocess the dataset
# Remove any rows with missing values
data = data.dropna()

# Split data into training set (first 500 samples)
# Reshape to column vectors as required by the model
train_input = np.array(data.x[0:500]).reshape(500, 1)  # Feature column 'x'
train_output = np.array(data.y[0:500]).reshape(500, 1)  # Target column 'y'

# Split data into test set (remaining samples)
# Note: There's a minor issue in the size (199 vs 200 samples)
test_input = np.array(data.x[500:700]).reshape(199, 1)  # Feature column 'x' for testing
test_output = np.array(data.y[500:700]).reshape(199, 1)  # Target column 'y' for testing

# Create and train the linear regression model
# Initialize model instance
linear_reg = LinearRegression()

# Train the model with the following hyperparameters:
# - Learning rate: 0.0001 (small value to ensure stable convergence)
# - Iterations: 20 (number of gradient descent steps)
# The training process will also create and save an animation showing the regression line fitting
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)