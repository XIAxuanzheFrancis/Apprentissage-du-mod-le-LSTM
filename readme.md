# We experiment with RNN, LSTM and GRU deep learning algorithms to predict indoor air quality

## Explanation of the document
1. **rnn_lstm_gru_lab.ipynb:** Predicting air quality using three deep learning algorithms RNN LSTM GRU on laboratoire.csv file. 

2. **room_analyse**: Analyze the air quality index and the correlation of each air detection parameter using the one_room_apartement.csv file

3. **lstm_room**: Predicting parameters for air detection on one_room_apartement.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)

4. **lstm_room_input2**: Predicting parameters for air detection on one_room_apartement.csv file using lstm deep learning algorithm(**input:** pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, oxygen, tvoc, co2, dewpt

5. **lstm_room_input3**: Predicting parameters for air detection on one_room_apartement.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, tvoc, co2, dewpt

6. **lstm_room_m1(class 2)**: Predicting parameters for air detection on one_room_apartement.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)---Unlike the previous one, here a simple air quality judgment algorithm is used to determine only good and bad air quality

7. **lab_analyse**： Analyze the air quality index and the correlation of each air detection parameter using the laboratory.csv file

8. **lstm_lab**: Predicting parameters for air detection on laboratory.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)

9. **lstm_lab_input2**: Predicting parameters for air detection on laboratory.csv file using lstm deep learning algorithm(**input:** pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, oxygen, tvoc, co2, dewpt

10. **lstm_lab_input3**: Predicting parameters for air detection on laboratory.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, tvoc, co2, dewpt

11. **lstm_lab_m1(class 2)**: Predicting parameters for air detection on one_room_apartement.csv file using lstm deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)---Unlike the previous one, here a simple air quality judgment algorithm is used to determine only good and bad air quality

12. **rnn_room**: Predicting parameters for air detection on one_room_apartement.csv file using rnn deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)

13. **rnn_room_input2**：Predicting parameters for air detection on one_room_apartement.csv file using rnn deep learning algorithm(**input:** pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, oxygen, tvoc, co2, dewpt

14. **rnn_room_input3**：Predicting parameters for air detection on laboratory.csv file using rnn deep learning algorithm(**input:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, tvoc, co2, dewpt

15. **rnn_lab**: Predicting parameters for air detection on laboratory.csv file using rnn deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)

16. **gru_room**: Predicting parameters for air detection on one_room_apartement.csv file using gru deep learning algorithm(**input:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, co2, so2, no2, pm1, dewpt, tvoc, o3, sound, pm2_5)

17. **gru_room_input2**: Predicting parameters for air detection on one_room_apartement.csv file using gru deep learning algorithm(**input:** pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, oxygen, tvoc, co2, dewpt

18. **gru_room_input3**: Predicting parameters for air detection on laboratory.csv file using gru deep learning algorithm(**input:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5, temperature, pressure, humidity **output:** oxygen, pm10, co, so2, no2, o3, sound, pm2_5) sans pm1, tvoc, co2, dewpt

19. **test1*: Analysis of data provided by the air descartes project

20. newIdea: The files in the newIdea folder are modified and optimized to correspond to the models in the corresponding files of the same name in the main directory.Mainly, the output removes a few detection parameters, and the output still predicts the removal of the detection parameters


Algorithms used to calculate the air index
1. https://www.kaggle.com/code/aliuoa/2023-indoor-air-quality-dataset-germany
2. https://www.kaggle.com/code/rohanrao/calculating-aqi-air-quality-index-tutorial