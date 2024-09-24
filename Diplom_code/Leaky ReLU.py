# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
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
# def leaky_relu(x, alpha=0.01):
#     return np.where(x > 0, x, x * alpha)
#
# def leaky_relu_derivative(x, alpha=0.01):
#     return np.where(x > 0, 1, alpha)
#
# def train_model():
#     sensors, conditions = load_sensor_data()
#
#     # Разделение данных на тренировочную и тестовую выборки
#     sensors_train, sensors_test, conditions_train, conditions_test = train_test_split(sensors, conditions, test_size=0.2, random_state=42)
#
#     input_size = sensors_train.shape[1]
#     hidden_size1 = 13
#     output_size = 1
#
#     weights_input_to_hidden1 = np.random.normal(0, 0.1, (hidden_size1, input_size))
#     weights_hidden1_to_output = np.random.normal(0, 0.1, (output_size, hidden_size1))
#
#     bias_input_to_hidden1 = np.zeros((hidden_size1, 1))
#     bias_hidden_to_output = np.zeros((output_size, 1))
#
#     epochs = 20
#     batch_size = 128
#     learning_rate = 0.001
#     beta1 = 0.9
#     beta2 = 0.999
#     epsilon = 1e-8
#
#     m_wih1 = np.zeros_like(weights_input_to_hidden1)
#     m_who = np.zeros_like(weights_hidden1_to_output)
#
#     v_wih1 = np.zeros_like(weights_input_to_hidden1)
#     v_who = np.zeros_like(weights_hidden1_to_output)
#
#     m_bih1 = np.zeros_like(bias_input_to_hidden1)
#     m_bho = np.zeros_like(bias_hidden_to_output)
#
#     v_bih1 = np.zeros_like(bias_input_to_hidden1)
#     v_bho = np.zeros_like(bias_hidden_to_output)
#
#     t = 0
#     loss_history = []
#     test_loss_history = []
#     accuracy_history = []
#
#     for epoch in range(epochs):
#         permutation = np.random.permutation(sensors_train.shape[0])
#         sensors_shuffled = sensors_train[permutation]
#         conditions_shuffled = conditions_train[permutation]
#
#         epoch_loss = 0
#
#         for i in range(0, sensors_train.shape[0], batch_size):
#             t += 1
#             x_batch = sensors_shuffled[i:i + batch_size].T
#             y_batch = conditions_shuffled[i:i + batch_size].T
#
#             # Forward pass
#             hidden1_raw = np.dot(weights_input_to_hidden1, x_batch) + bias_input_to_hidden1
#             hidden1 = leaky_relu(hidden1_raw)
#             output_raw = np.dot(weights_hidden1_to_output, hidden1) + bias_hidden_to_output
#             output = 1 / (1 + np.exp(-output_raw))  # Активация сигмоиды на выходе
#
#             # Вычисление потерь
#             loss = np.mean((output - y_batch) ** 2)
#             epoch_loss += loss
#
#             # Обратное распространение (Backpropagation)
#             d_output = output - y_batch
#             d_weights_hidden1_to_output = np.dot(d_output, hidden1.T)
#             d_bias_hidden_to_output = np.sum(d_output, axis=1, keepdims=True)
#
#             d_hidden1 = np.dot(weights_hidden1_to_output.T, d_output) * leaky_relu_derivative(hidden1_raw)
#             d_weights_input_to_hidden1 = np.dot(d_hidden1, x_batch.T)
#             d_bias_input_to_hidden1 = np.sum(d_hidden1, axis=1, keepdims=True)
#
#             # Обновление параметров с использованием Adam
#             for param, dparam, m, v in zip(
#                     [weights_input_to_hidden1, weights_hidden1_to_output,
#                      bias_input_to_hidden1, bias_hidden_to_output],
#                     [d_weights_input_to_hidden1, d_weights_hidden1_to_output,
#                      d_bias_input_to_hidden1, d_bias_hidden_to_output],
#                     [m_wih1, m_who, m_bih1, m_bho],
#                     [v_wih1, v_who, v_bih1, v_bho]):
#                 m[:] = beta1 * m + (1 - beta1) * dparam
#                 v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
#                 m_hat = m / (1 - beta1 ** t)
#                 v_hat = v / (1 - beta2 ** t)
#                 param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
#
#         # Проверка на тестовой выборке
#         x_test = sensors_test.T
#         y_test = conditions_test.T
#
#         hidden1_raw_test = np.dot(weights_input_to_hidden1, x_test) + bias_input_to_hidden1
#         hidden1_test = leaky_relu(hidden1_raw_test)
#         output_raw_test = np.dot(weights_hidden1_to_output, hidden1_test) + bias_hidden_to_output
#         output_test = 1 / (1 + np.exp(-output_raw_test))  # Активация сигмоиды на выходе
#
#         test_loss = np.mean((output_test - y_test) ** 2)
#         test_predictions = (output_test > 0.5).astype(int)
#         accuracy = accuracy_score(y_test.flatten(), test_predictions.flatten())
#
#         loss_history.append(epoch_loss / (sensors_train.shape[0] // batch_size))
#         test_loss_history.append(test_loss)
#         accuracy_history.append(accuracy)
#
#         print(f'Epoch {epoch + 1}, Loss: {loss_history[-1]}, Test Loss: {test_loss}, Test Accuracy: {accuracy}')
#
#     # Визуализация обучения
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(1, epochs + 1), loss_history, label='Training Loss', marker='o')
#     plt.plot(range(1, epochs + 1), test_loss_history, label='Test Loss', marker='x')
#     plt.title('Training and Test Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(1, epochs + 1), accuracy_history, label='Test Accuracy', marker='s')
#     plt.title('Test Accuracy Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     # Сохранение обученной модели
#     np.savez('trained_model.npz',
#              weights_input_to_hidden1=weights_input_to_hidden1,
#              weights_hidden1_to_output=weights_hidden1_to_output,
#              bias_input_to_hidden1=bias_input_to_hidden1,
#              bias_hidden_to_output=bias_hidden_to_output)
#
#     print("Модель обучена и сохранена в 'trained_model.npz'")
#
# def predict(model, input_data):
#     weights_input_to_hidden1 = model['weights_input_to_hidden1']
#     weights_hidden1_to_output = model['weights_hidden1_to_output']
#     bias_input_to_hidden1 = model['bias_input_to_hidden1']
#     bias_hidden_to_output = model['bias_hidden_to_output']
#
#     x = input_data.T
#     hidden1_raw = np.dot(weights_input_to_hidden1, x) + bias_input_to_hidden1
#     hidden1 = leaky_relu(hidden1_raw)
#     output_raw = np.dot(weights_hidden1_to_output, hidden1) + bias_hidden_to_output
#     output = 1 / (1 + np.exp(-output_raw))
#
#     return (output > 0.5).astype(int)
#
# if __name__ == "__main__":
#     train_model()
#
#     # Пример использования предсказания на новых данных
#     with np.load('trained_model.npz') as model:
#         new_input = np.array([[500, 200, 50]]).astype('float32')
#         new_input /= np.max(new_input, axis=0)
#         prediction = predict(model, new_input)
#         print(f"Предсказание для {new_input}: {prediction}")




