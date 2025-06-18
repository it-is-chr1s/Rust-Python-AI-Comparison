First run
Training was done on one gpu core.
Loading the datasets needed 2628.60703 ms
Balancing the datasets needed 193.62307 ms
Converting to One Hot Encoding needed 72.77107 ms
Y_train_one-hot-encoding: (101140, 196)
Y_test_one-hot-encoding: (33769, 196)
Splitting test set into validation- and test set needed 0.06294 ms
Training set shape : (101140, 100, 100, 3)
Validation set shape : (7000, 100, 100, 3)
Test set shape : (26769, 100, 100, 3)
1st training image shape (100, 100, 3)
Data Preprocessing needed 51374.90416 ms
Training CNN needed 454983.82950 ms
Saving Model needed 43.29395 ms
837/837 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step
Prediction needed 8903.92780 ms
Logs saved to ./artifacts/time_log/cnn_time_log20250514-192055.txt
837/837 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9807 - loss: 0.1328
Test accuracy: 0.980201005935669
