import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from TimeMeasuring import TimeMeasuring

def main():
    tm = TimeMeasuring("./time_log/cnn_time_log_")

    melbourne_file_path = '../../datasets/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    tm.took("Loading the dataset")

    print(melbourne_data.describe())
    tm.took("Describe for whole dataset")

    melbourne_data = melbourne_data.dropna(axis=0)
    tm.took("Drop missing values")

    y = melbourne_data.Price
    tm.took("Get single column price")

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    tm.took("Selecting features")

    print(X.describe())
    tm.took("Describe for modified dataset")

    print(X.head())
    tm.took("Printing head")

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    tm.took("Splitting in training and validation dataset")

    melbourne_model = DecisionTreeRegressor(random_state=1)

    melbourne_model.fit(train_X, train_y)
    tm.took("Training")

    val_predictions = melbourne_model.predict(val_X)
    tm.took("Prediction")

    top5_features = val_X.head(5)
    top5_predictions = val_predictions[:5]
    top5_real = val_y.head(5)
    print("The 5 validation features and their predictions:")
    for i in range(5):
        print("Features: ", top5_features.iloc[i].to_dict())
        print("Prediction: ", top5_predictions[i])
        print("Real: ", top5_real.iloc[i])
        print("---")

    tm.reset()
    mae = mean_absolute_error(val_y, val_predictions)
    tm.took("Calculating MAE")
    print("Mean absolute error: ", mae)

    tm.save_log()

if __name__ == "__main__":
    main()