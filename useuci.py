from energy_comsumption_analyzer import *
from data import *


# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    analyzer = EnergyConsumptionAnalyzer()


    train_path = 'data/preprocessed_data/preprocessed_train.csv'
    test_path = 'data/preprocessed_data/preprocessed_test.csv'
    loaded_data = analyzer.load_data(train_path, test_path)

    analyzer.preprocess_data()

    # Train LSTM model
    analyzer.train_pytorch_model('lstm', prediction_horizon=90 ,epochs=1001)

    analyzer.evaluate_models(prediction_horizon=90)

    # Make predictions
    predictions, dates = analyzer.predict_future('lstm', steps=90)