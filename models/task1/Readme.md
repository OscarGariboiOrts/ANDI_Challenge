12 models were trained for Task1. We built trajectory length dependant models for lengths from:
  - 10 to 20
  - 20 to 30
  - 30 to 40
  - 40 to 50
  - 50 to 100
  - 100 to 200
  - 200 to 300
  - 300 to 400
  - 400 to 500
  - 500 to 600
  - 600 to 800
  - 800 to 1000

All models used for Task1 were trained using the same architecture, which consists of two stacked Conv1D layers followed by 4 Bidirectional LSTM layer with Dropout layers between them, followed by one Dense layer.

Model architecture for trajectory length between 10 and 20 is show in the following picture:

![task1_len_10_20 h5](https://user-images.githubusercontent.com/29183729/112234898-d1760b00-8c3d-11eb-9514-b9606d7205f8.png)
