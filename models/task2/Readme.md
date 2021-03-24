Three models were trained for Task2, one per subtask (1D, 2D and 3D trajectories, respectively).

1D trajectories.
  - 4 million trajectories of variable length (from 10 to 1000) were generated and used to train the model.
  - model architecture consists of two stacked Conv1D layers, followed by two Bidirectional LSTM layer with a Dropout layer between them, and having a Dense layer to produce the classification prediction.
 
 Model architecture shown in the following picture:
 
 ![task2_dim1 h5](https://user-images.githubusercontent.com/29183729/112235621-5281d200-8c3f-11eb-92b9-0ce6b758f02f.png)
 
 2D trajectories:
  - 2 million trajectories of variable length (from 10 to 1000) were generated and used to train the model.
  - model architecture consists of two stacked Conv1D layers, followed by three Bidirectional LSTM layer with a Dropout layer between them, and having two Dense layers to produce the classification prediction.
 
 Model architecture shown in the following picture:

![task2_dim2 h5](https://user-images.githubusercontent.com/29183729/112235771-9c6ab800-8c3f-11eb-8a34-d45904fa4456.png)

3D trajectories:
- 2 million trajectories of variable length (from 10 to 1000) were generated and used to train the model.
  - model architecture consists of two stacked Conv1D layers, followed by three Bidirectional LSTM layer with a Dropout layer between them, and having two Dense layers to produce the classification prediction.
 
 Model architecture shown in the following picture:

![task2_dim3 h5](https://user-images.githubusercontent.com/29183729/112235861-c1f7c180-8c3f-11eb-8882-5dbbf8e34ae4.png)
