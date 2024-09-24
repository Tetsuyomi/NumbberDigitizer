# import numpy as np
#
# def load_model(filepath):
#     with np.load(filepath) as data:
#         weights_input_to_hidden1 = data['weights_input_to_hidden1']
#         weights_hidden1_to_hidden2 = data['weights_hidden1_to_hidden2']
#         weights_hidden2_to_output = data['weights_hidden2_to_output']
#         bias_input_to_hidden1 = data['bias_input_to_hidden1']
#         bias_hidden1_to_hidden2 = data['bias_hidden1_to_hidden2']
#         bias_hidden_to_output = data['bias_hidden_to_output']
#     return (weights_input_to_hidden1, weights_hidden1_to_hidden2, weights_hidden2_to_output,
#             bias_input_to_hidden1, bias_hidden1_to_hidden2, bias_hidden_to_output)
#
# def relu(x):
#     return np.maximum(0, x)
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def predict(model, x):
#     weights_input_to_hidden1, weights_hidden1_to_hidden2, weights_hidden2_to_output, \
#     bias_input_to_hidden1, bias_hidden1_to_hidden2, bias_hidden_to_output = model
#
#     hidden1_raw = np.dot(weights_input_to_hidden1, x.T) + bias_input_to_hidden1
#     hidden1 = relu(hidden1_raw)
#     hidden2_raw = np.dot(weights_hidden1_to_hidden2, hidden1) + bias_hidden1_to_hidden2
#     hidden2 = relu(hidden2_raw)
#     output_raw = np.dot(weights_hidden2_to_output, hidden2) + bias_hidden_to_output
#     output = sigmoid(output_raw)
#     return output.T
#
# def main():
#     model = load_model('trained_model.npz')
#     print("Введите три значения датчиков:")
#     sensor_1 = float(input("Значение датчика 1: "))
#     sensor_2 = float(input("Значение датчика 2: "))
#     sensor_3 = float(input("Значение датчика 3: "))
#
#     # Нормализация входных данных
#     input_data = np.array([[sensor_1, sensor_2, sensor_3]], dtype='float32')
#     input_data /= np.max(input_data, axis=1)
#
#     prediction = predict(model, input_data)
#     result = (prediction > 0.5).astype(int)
#     print(f'Предсказание: {result[0][0]}')
#
# if __name__ == "__main__":
#     main()








# import numpy as np
#
# def load_model(filepath):
#     with np.load(filepath) as data:
#         weights_input_to_hidden1 = data['weights_input_to_hidden1']
#         weights_hidden1_to_output = data['weights_hidden1_to_output']
#         bias_input_to_hidden1 = data['bias_input_to_hidden1']
#         bias_hidden_to_output = data['bias_hidden_to_output']
#     return (weights_input_to_hidden1, weights_hidden1_to_output,
#             bias_input_to_hidden1, bias_hidden_to_output)
#
# def relu(x):
#     return np.maximum(0, x)
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def predict(model, x):
#     weights_input_to_hidden1, weights_hidden1_to_output, \
#     bias_input_to_hidden1, bias_hidden_to_output = model
#
#     hidden1_raw = np.dot(weights_input_to_hidden1, x.T) + bias_input_to_hidden1
#     hidden1 = relu(hidden1_raw)
#     output_raw = np.dot(weights_hidden1_to_output, hidden1) + bias_hidden_to_output
#     output = sigmoid(output_raw)
#     return output.T
#
# def main():
#     model = load_model('trained_model.npz')
#     print("Введите три значения датчиков:")
#     sensor_1 = float(input("Значение датчика 1: "))
#     sensor_2 = float(input("Значение датчика 2: "))
#     sensor_3 = float(input("Значение датчика 3: "))
#
#     # Нормализация входных данных
#     input_data = np.array([[sensor_1, sensor_2, sensor_3]], dtype='float32')
#     input_data /= np.max(input_data, axis=1)
#
#     prediction = predict(model, input_data)
#     result = (prediction > 0.5).astype(int)
#     print(f'Предсказание: {result[0][0]}')
#
# if __name__ == "__main__":
#     main()


import numpy as np
import keyboard  # Библиотека для отслеживания нажатий клавиш

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_model(filepath):
    with np.load(filepath) as data:
        weights_input_to_hidden1 = data['weights_input_to_hidden1']
        weights_hidden1_to_output = data['weights_hidden1_to_output']
        bias_input_to_hidden1 = data['bias_input_to_hidden1']
        bias_hidden_to_output = data['bias_hidden_to_output']
    return (weights_input_to_hidden1, weights_hidden1_to_output,
            bias_input_to_hidden1, bias_hidden_to_output)




def predict(model, x):
    weights_input_to_hidden1, weights_hidden1_to_output, \
        bias_input_to_hidden1, bias_hidden_to_output = model

    hidden1_raw = np.dot(weights_input_to_hidden1, x.T) + bias_input_to_hidden1
    hidden1 = tanh(hidden1_raw)
    output_raw = np.dot(weights_hidden1_to_output, hidden1) + bias_hidden_to_output
    output = sigmoid(output_raw)
    return output.T


def get_sensor_value(sensor_name):
    while True:
        value = input(f"Значение {sensor_name}: ")
        if value.isdigit():
            return float(value)
        else:
            print("Введенное значение не является целым числом. Пожалуйста, попробуйте снова.")


def main():
    model = load_model('trained_model_5epoh.npz')

    while True:
        if keyboard.is_pressed('q'):  # Проверка нажатия клавиши "Q"
            print("Выход из программы.")
            break

        print("Введите три значения датчиков:")
        sensor_1 = get_sensor_value("датчика 1")
        sensor_2 = get_sensor_value("датчика 2")
        sensor_3 = get_sensor_value("датчика 3")

        # Нормализация входных данных
        input_data = np.array([[sensor_1, sensor_2, sensor_3]], dtype='float32')
        input_data /= np.max(input_data, axis=1)

        prediction = predict(model, input_data)
        result = (prediction > 0.5).astype(int)
        print(f'Предсказание: {result[0][0]}')


if __name__ == "__main__":
    main()

