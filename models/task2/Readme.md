Three models were trained for Task2, one per subtask (1D, 2D and 3D trajectories, respectively).

1D trajectories.
  - 4 million trajectories of variable length (from 10 to 1000) were generated and used to train the model.
  - model architecture consists of two stacked Conv1D layers, followed by two Bidirectional LSTM layer with a Dropout layer between them, and having a Dense layer to produce the classification prediction.
 
 Model architecture shown in the following picture:
 
 
