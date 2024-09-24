





#Код без оптимизации с одим скрытым слоем






# import numpy as np
# import matplotlib.pyplot as plt
#
# def load_sensor_data() -> object:
#     # Загрузка данных с датчиков
#     with np.load('sensor_data.npz') as f:
#         x_train = f['x_train'].astype('float32')  # Данные с датчиков
#         y_train = f['y_train']  # Состояние оборудования: 0 - норма, 1 - неисправность
#
#         # Нормализация данных
#         x_train /= np.max(x_train, axis=0)
#
#         return x_train, y_train
#
# sensors, conditions = load_sensor_data()
#
# weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (50, sensors.shape[1]))  # Веса для входного слоя
# weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (1, 50))  # Веса для скрытого слоя
#
# bias_input_to_hidden = np.zeros((50, 1))
# bias_hidden_to_output = np.zeros((1, 1))
#
# epochs = 3
# e_loss = 0
# e_correct = 0
# learning_rate = 0.01
#
# for epoch in range(epochs):
#     print(f'Epoch №{epoch}')
#
#     for sensor, condition in zip(sensors, conditions):
#         sensor = np.reshape(sensor, (-1, 1))
#         condition = np.reshape(condition, (-1, 1))
#
#         hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ sensor
#         hidden = 1 / (1 + np.exp(-hidden_raw))  # Сигмоидная функция активации
#
#         output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
#         output = 1 / (1 + np.exp(-output_raw))  # Сигмоидная функция активации
#
#         e_loss += 1/ len(output) * np.sum((output - condition)**2, axis=0)
#         e_correct += int((output > 0.5) == condition)
#
#         delta_output = output - condition
#         weights_hidden_to_output += - learning_rate * delta_output @ np.transpose(hidden)
#         bias_hidden_to_output += -learning_rate * delta_output
#
#         delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
#         weights_input_to_hidden += - learning_rate * delta_hidden @ np.transpose(sensor)
#         bias_input_to_hidden += -learning_rate * delta_hidden
#
#     print(f'Loss {round((e_loss[0]/ sensors.shape[0]) * 100, 3)}%')
#     print(f'Accuracy  {round((e_correct / sensors.shape[0]) * 100, 3)}%')
#     e_loss = 0
#     e_correct = 0
#
# # Используем обученные веса и смещения в вашей нейронной сети для предсказания
# test_sensor = np.random.normal(size=(sensors.shape[1],))  # Пример новых данных с датчиков
# test_sensor = np.reshape(test_sensor, (-1, 1))
#
# hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ test_sensor
# hidden = 1 / (1 + np.exp(-hidden_raw))
#
# output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
# output = 1 / (1 + np.exp(-output_raw))
#
# predicted_condition = "неисправность" if output > 0.5 else "норма"
# print(f'Состояние оборудования: {predicted_condition}')
#
#
#
#
#
#
#
#
# #Код после оптимизации
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
#
#
# def load_sensor_data() -> object:
#     with np.load('sensor_data.npz') as f:
#         x_train = f['x_train'].astype('float32')
#         y_train = f['y_train']
#         x_train /= np.max(x_train, axis=0)  # Нормализация
#         return x_train, y_train
#
#
# def relu(x):
#     return np.maximum(0, x)
#
#
# def relu_derivative(x):
#     return (x > 0).astype(x.dtype)
#
#
# sensors, conditions = load_sensor_data()
#
# input_size = sensors.shape[1]
# hidden_size = 50
# output_size = 1
#
# weights_input_to_hidden = np.random.normal(0, 0.1, (hidden_size, input_size))
# weights_hidden_to_output = np.random.normal(0, 0.1, (output_size, hidden_size))
#
# bias_input_to_hidden = np.zeros((hidden_size, 1))
# bias_hidden_to_output = np.zeros((output_size, 1))
#
# epochs = 10
# batch_size = 64
# learning_rate = 0.001
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
#
# m_wih = np.zeros_like(weights_input_to_hidden)
# m_who = np.zeros_like(weights_hidden_to_output)
# v_wih = np.zeros_like(weights_input_to_hidden)
# v_who = np.zeros_like(weights_hidden_to_output)
#
# m_bih = np.zeros_like(bias_input_to_hidden)
# m_bho = np.zeros_like(bias_hidden_to_output)
# v_bih = np.zeros_like(bias_input_to_hidden)
# v_bho = np.zeros_like(bias_hidden_to_output)
#
# t = 0
#
# for epoch in range(epochs):
#     # Shuffle the dataset
#     permutation = np.random.permutation(sensors.shape[0])
#     sensors_shuffled = sensors[permutation]
#     conditions_shuffled = conditions[permutation]
#
#     for i in range(0, sensors.shape[0], batch_size):
#         t += 1
#         x_batch = sensors_shuffled[i:i + batch_size].T
#         y_batch = conditions_shuffled[i:i + batch_size].T
#
#         # Forward pass
#         hidden_raw = np.dot(weights_input_to_hidden, x_batch) + bias_input_to_hidden
#         hidden = relu(hidden_raw)
#         output_raw = np.dot(weights_hidden_to_output, hidden) + bias_hidden_to_output
#         output = 1 / (1 + np.exp(-output_raw))  # Sigmoid activation for output
#
#         # Loss calculation
#         loss = np.mean((output - y_batch) ** 2)
#
#         # Backpropagation
#         d_output = output - y_batch
#         d_weights_hidden_to_output = np.dot(d_output, hidden.T)
#         d_bias_hidden_to_output = np.sum(d_output, axis=1, keepdims=True)
#
#         d_hidden = np.dot(weights_hidden_to_output.T, d_output) * relu_derivative(hidden_raw)
#         d_weights_input_to_hidden = np.dot(d_hidden, x_batch.T)
#         d_bias_input_to_hidden = np.sum(d_hidden, axis=1, keepdims=True)
#
#         # Adam optimization for weights and biases
#         # Update weights and biases using Adam optimizer
#         for param, dparam, m, v in zip(
#                 [weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output],
#                 [d_weights_input_to_hidden, d_weights_hidden_to_output, d_bias_input_to_hidden,
#                  d_bias_hidden_to_output],
#                 [m_wih, m_who, m_bih, m_bho],
#                 [v_wih, v_who, v_bih, v_bho]):
#             m[:] = beta1 * m + (1 - beta1) * dparam
#             v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
#             m_hat = m / (1 - beta1 ** t)
#             v_hat = v / (1 - beta2 ** t)
#             param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
#
#     print(f'Epoch {epoch + 1}, Loss: {loss}')
#
# # Предсказание (можно добавить, если требуется демонстрация)
# test_index = np.random.randint(0, sensors.shape[0])
# test_sensor = sensors[test_index]
# hidden_raw = np.dot(weights_input_to_hidden, test_sensor) + bias_input_to_hidden
# hidden = relu(hidden_raw)
# output_raw = np.dot(weights_hidden_to_output, hidden) + bias_hidden_to_output
# output = 1 / (1 + np.exp(-output_raw))
# predicted_condition = "неисправность" if output > 0.5 else "норма"
# print(f'Предсказанное состояние для тестового примера: {predicted_condition}')












#Код после опптимизации с двумя скрытыми слоями














import numpy as np

#Загрузка данных
# def load_sensor_data() -> object:
#     with np.load('sensor_data.npz') as f:
#         x_train = f['x_train'].astype('float32')
#         y_train = f['y_train']
#         x_train /= np.max(x_train, axis=0)  # Нормализация
#         return x_train, y_train
# import numpy as np
#
#
# def load_sensor_data() -> tuple:
#     # Загрузка данных из первого файла NPZ
#     with np.load('data_1.npz') as f1:
#         x_train_1 = f1['x_train_1'].astype('float32')
#         x_train_2 = f1['x_train_2'].astype('float32')
#         x_train_3 = f1['x_train_3'].astype('float32')
#
#     # Загрузка данных из второго файла NPZ
#     with np.load('data_2.npz') as f2:
#         y_train = f2['y_train']
#
#     # Объединение данных x_train из первого файла в один массив
#     x_train = np.column_stack((x_train_1, x_train_2, x_train_3))
#
#     # Нормализация данных x_train
#     x_train /= np.max(x_train, axis=0)  # Нормализация
#
#     return x_train, y_train
#
#
# # Пример использования
# x_train, y_train = load_sensor_data()
# print(x_train.shape, y_train.shape)
#
#
# #Инициализация функции ReLU
# def relu(x):
#     return np.maximum(0, x)
#
# #Направление функции ReLU
# def relu_derivative(x):
#     return (x > 0).astype(x.dtype)
#
#
# sensors, conditions = load_sensor_data()
#
# input_size = sensors.shape[1]
# hidden_size1 = 50
# hidden_size2 = 30  # Второй скрытый слой
# output_size = 1
#
# weights_input_to_hidden1 = np.random.normal(0, 0.1, (hidden_size1, input_size))
# weights_hidden1_to_hidden2 = np.random.normal(0, 0.1,(hidden_size2, hidden_size1))  # Веса между первым и вторым скрытым слоем
# weights_hidden2_to_output = np.random.normal(0, 0.1, (output_size, hidden_size2))
#
# bias_input_to_hidden1 = np.zeros((hidden_size1, 1))
# bias_hidden1_to_hidden2 = np.zeros((hidden_size2, 1))  # Смещение для второго скрытого слоя
# bias_hidden_to_output = np.zeros((output_size, 1))
#
# epochs = 10
# batch_size = 64
# learning_rate = 0.001
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
#
# m_wih1 = np.zeros_like(weights_input_to_hidden1)
# m_wh1h2 = np.zeros_like(weights_hidden1_to_hidden2)
# m_wh2o = np.zeros_like(weights_hidden2_to_output)
#
# v_wih1 = np.zeros_like(weights_input_to_hidden1)
# v_wh1h2 = np.zeros_like(weights_hidden1_to_hidden2)
# v_wh2o = np.zeros_like(weights_hidden2_to_output)
#
# m_bih1 = np.zeros_like(bias_input_to_hidden1)
# m_bh1h2 = np.zeros_like(bias_hidden1_to_hidden2)
# m_bho = np.zeros_like(bias_hidden_to_output)
#
# v_bih1 = np.zeros_like(bias_input_to_hidden1)
# v_bh1h2 = np.zeros_like(bias_hidden1_to_hidden2)
# v_bho = np.zeros_like(bias_hidden_to_output)
#
# t = 0
#
# for epoch in range(epochs):
#     permutation = np.random.permutation(sensors.shape[0])
#     sensors_shuffled = sensors[permutation]
#     conditions_shuffled = conditions[permutation]
#
#     for i in range(0, sensors.shape[0], batch_size):
#         t += 1
#         x_batch = sensors_shuffled[i:i + batch_size].T
#         y_batch = conditions_shuffled[i:i + batch_size].T
#
#         # Forward pass
#         hidden1_raw = np.dot(weights_input_to_hidden1, x_batch) + bias_input_to_hidden1
#         hidden1 = relu(hidden1_raw)
#         hidden2_raw = np.dot(weights_hidden1_to_hidden2, hidden1) + bias_hidden1_to_hidden2
#         hidden2 = relu(hidden2_raw)
#         output_raw = np.dot(weights_hidden2_to_output, hidden2) + bias_hidden_to_output
#         output = 1 / (1 + np.exp(-output_raw))  #Активация сигмоиды на выход
#
#         #Вычисление потерь
#         loss = np.mean((output - y_batch) ** 2)
#
#         #Обратное распространение(Backpropagation)
#         d_output = output - y_batch
#         d_weights_hidden2_to_output = np.dot(d_output, hidden2.T)
#         d_bias_hidden_to_output = np.sum(d_output, axis=1, keepdims=True)
#
#         d_hidden2 = np.dot(weights_hidden2_to_output.T, d_output) * relu_derivative(hidden2_raw)
#         d_weights_hidden1_to_hidden2 = np.dot(d_hidden2, hidden1.T)
#         d_bias_hidden1_to_hidden2 = np.sum(d_hidden2, axis=1, keepdims=True)
#
#         d_hidden1 = np.dot(weights_hidden1_to_hidden2.T, d_hidden2) * relu_derivative(hidden1_raw)
#         d_weights_input_to_hidden1 = np.dot(d_hidden1, x_batch.T)
#         d_bias_input_to_hidden1 = np.sum(d_hidden1, axis=1, keepdims=True)
#
#         #Обновление параметров с использованием Adam
#         for param, dparam, m, v in zip([weights_input_to_hidden1, weights_hidden1_to_hidden2, weights_hidden2_to_output,
#                                         bias_input_to_hidden1, bias_hidden1_to_hidden2, bias_hidden_to_output],
#                                        [d_weights_input_to_hidden1, d_weights_hidden1_to_hidden2,
#                                         d_weights_hidden2_to_output,
#                                         d_bias_input_to_hidden1, d_bias_hidden1_to_hidden2, d_bias_hidden_to_output],
#                                        [m_wih1, m_wh1h2, m_wh2o, m_bih1, m_bh1h2, m_bho],
#                                        [v_wih1, v_wh1h2, v_wh2o, v_bih1, v_bh1h2, v_bho]):
#             m[:] = beta1 * m + (1 - beta1) * dparam
#             v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
#             m_hat = m / (1 - beta1 ** t)
#             v_hat = v / (1 - beta2 ** t)
#             param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
#
#     print(f'Epoch {epoch + 1}, Loss: {loss}')

# Пример использования обученной модели для прогнозирования состояния оборудования может быть добавлен по необходимости.














# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Загрузка данных с датчиков
# def load_sensor_data() -> object:
#     with np.load('sensor_data.npz') as f:
#         x_train = f['x_train'].astype('float32')
#         y_train = f['y_train']
#         x_train /= np.max(x_train, axis=0)  # Нормализация
#         return x_train, y_train
#
#
# def relu(x):
#     return np.maximum(0, x)
#
#
# def relu_derivative(x):
#     return (x > 0).astype(x.dtype)
#
#
# # Инициализация данных
# sensors, conditions = load_sensor_data()
# input_size = sensors.shape[1]
# hidden_size1 = 50
# hidden_size2 = 30  # Второй скрытый слой
# output_size = 1
#
# # Инициализация весов и смещений
# weights_input_to_hidden1 = np.random.normal(0, 0.1, (hidden_size1, input_size))
# weights_hidden1_to_hidden2 = np.random.normal(0, 0.1, (hidden_size2, hidden_size1))
# weights_hidden2_to_output = np.random.normal(0, 0.1, (output_size, hidden_size2))
# bias_input_to_hidden1 = np.zeros((hidden_size1, 1))
# bias_hidden1_to_hidden2 = np.zeros((hidden_size2, 1))
# bias_hidden_to_output = np.zeros((output_size, 1))
#
# epochs = 10
# batch_size = 64
# learning_rate = 0.001
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
#
# # Для оптимизации Adam
# m_wih1 = np.zeros_like(weights_input_to_hidden1)
# m_wh1h2 = np.zeros_like(weights_hidden1_to_hidden2)
# m_wh2o = np.zeros_like(weights_hidden2_to_output)
# v_wih1 = np.zeros_like(weights_input_to_hidden1)
# v_wh1h2 = np.zeros_like(weights_hidden1_to_hidden2)
# v_wh2o = np.zeros_like(weights_hidden2_to_output)
# m_bih1 = np.zeros_like(bias_input_to_hidden1)
# m_bh1h2 = np.zeros_like(bias_hidden1_to_hidden2)
# m_bho = np.zeros_like(bias_hidden_to_output)
# v_bih1 = np.zeros_like(bias_input_to_hidden1)
# v_bh1h2 = np.zeros_like(bias_hidden1_to_hidden2)
# v_bho = np.zeros_like(bias_hidden_to_output)
#
# t = 0
# loss_history = []
#
# # Обучение модели
# for epoch in range(epochs):
#     permutation = np.random.permutation(sensors.shape[0])
#     sensors_shuffled = sensors[permutation]
#     conditions_shuffled = conditions[permutation]
#
#     epoch_loss = 0
#
#     for i in range(0, sensors.shape[0], batch_size):
#         t += 1
#         x_batch = sensors_shuffled[i:i + batch_size].T
#         y_batch = conditions_shuffled[i:i + batch_size].T
#
#         # Прямой проход
#         hidden1_raw = np.dot(weights_input_to_hidden1, x_batch) + bias_input_to_hidden1
#         hidden1 = relu(hidden1_raw)
#         hidden2_raw = np.dot(weights_hidden1_to_hidden2, hidden1) + bias_hidden1_to_hidden2
#         hidden2 = relu(hidden2_raw)
#         output_raw = np.dot(weights_hidden2_to_output, hidden2) + bias_hidden_to_output
#         output = 1 / (1 + np.exp(-output_raw))  # Сигмоидная функция для выхода
#
#         # Расчет потерь
#         loss = np.mean((output - y_batch) ** 2)
#         epoch_loss += loss
#
#         # Обратное распространение ошибки
#         d_output = output - y_batch
#         d_weights_hidden2_to_output = np.dot(d_output, hidden2.T)
#         d_bias_hidden_to_output = np.sum(d_output, axis=1, keepdims=True)
#         d_hidden2 = np.dot(weights_hidden2_to_output.T, d_output) * relu_derivative(hidden2_raw)
#         d_weights_hidden1_to_hidden2 = np.dot(d_hidden2, hidden1.T)
#         d_bias_hidden1_to_hidden2 = np.sum(d_hidden2, axis=1, keepdims=True)
#         d_hidden1 = np.dot(weights_hidden1_to_hidden2.T, d_hidden2) * relu_derivative(hidden1_raw)
#         d_weights_input_to_hidden1 = np.dot(d_hidden1, x_batch.T)
#         d_bias_input_to_hidden1 = np.sum(d_hidden1, axis=1, keepdims=True)
#
#         # Обновление параметров с использованием Adam
#         for param, dparam, m, v in zip(
#                 [weights_input_to_hidden1, weights_hidden1_to_hidden2, weights_hidden2_to_output, bias_input_to_hidden1,
#                  bias_hidden1_to_hidden2, bias_hidden_to_output],
#                 [d_weights_input_to_hidden1, d_weights_hidden1_to_hidden2, d_weights_hidden2_to_output,
#                  d_bias_input_to_hidden1, d_bias_hidden1_to_hidden2, d_bias_hidden_to_output],
#                 [m_wih1, m_wh1h2, m_wh2o, m_bih1, m_bh1h2, m_bho],
#                 [v_wih1, v_wh1h2, v_wh2o, v_bih1, v_bh1h2, v_bho]):
#             m[:] = beta1 * m + (1 - beta1) * dparam
#             v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
#             m_hat = m / (1 - beta1 ** t)
#             v_hat = v / (1 - beta2 ** t)
#             param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
#
#     loss_history.append(epoch_loss / (sensors.shape[0] // batch_size))
#     print(f'Epoch {epoch + 1}, Loss: {epoch_loss / (sensors.shape[0] // batch_size)}')
#
# # Визуализация обучения
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, epochs + 1), loss_history, label='Model with 2 Hidden Layers', marker='o')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Сравнение с другими моделями (например, с одной скрытым слоем)
# # Здесь предполагается, что у вас есть функция для обучения модели с одним скрытым слоем
# # и сохранение потерь в other_model_loss_history
#
# # Пример сравнения с другой моделью
# other_model_loss_history = [np.random.uniform(0.5, 1.5) for _ in range(epochs)]  # Имитация данных
#
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, epochs + 1), loss_history, label='Model with 2 Hidden Layers', marker='o')
# plt.plot(range(1, epochs + 1), other_model_loss_history, label='Model with 1 Hidden Layer', marker='x')
# plt.title('Comparison of Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt

# Assuming the data points from the provided images
# Image 1 data points
epochs_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_1 = [0.91, 0.911, 0.913, 0.912, 0.914, 0.92, 0.93, 0.945, 0.96, 0.98]

# Image 2 data points
epochs_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_2 = [0.8925, 0.906, 0.9075, 0.9085, 0.909, 0.9075, 0.908, 0.908, 0.909, 0.9085]

# Image 3 data points
epochs_3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_3 = [0.9, 0.905, 0.907, 0.908, 0.9085, 0.907, 0.9075, 0.908, 0.907, 0.9075]

plt.figure(figsize=(10, 5))

# Plot data from Image 1
plt.plot(epochs_1, accuracy_1, label='Tanh Accuracy', marker='s', color='blue')

# Plot data from Image 2
plt.plot(epochs_2, accuracy_2, label='Sigmoid Accuracy', marker='o', color='green')

# Plot data from Image 3
plt.plot(epochs_3, accuracy_3, label='ReLU Accuracy', marker='^', color='red')

# Adding titles and labels
plt.title('Combined Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the combined plot
plt.show()
