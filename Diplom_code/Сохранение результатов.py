import numpy as np

# Предполагаем, что у вас есть обученные веса и смещения:
# weights_input_to_hidden1, weights_hidden1_to_hidden2, weights_hidden2_to_output
# bias_input_to_hidden1, bias_hidden1_to_hidden2, bias_hidden_to_output

# Сохранение модели
def save_model_npz(weights, biases, filename='model.npz'):
    np.savez(filename, **weights, **biases)

# Загрузка модели
def load_model_npz(filename='model.npz'):
    data = np.load(filename)
    model = {
        'weights': {
            'w_input_to_hidden1': data['w_input_to_hidden1'],
            'w_hidden1_to_hidden2': data['w_hidden1_to_hidden2'],
            'w_hidden2_to_output': data['w_hidden2_to_output']
        },
        'biases': {
            'b_input_to_hidden1': data['b_input_to_hidden1'],
            'b_hidden1_to_hidden2': data['b_hidden1_to_hidden2'],
            'b_hidden_to_output': data['b_hidden_to_output']
        }
    }
    return model

# Объедините все веса и смещения в структуру для сохранения
weights = {
    'w_input_to_hidden1': weights_input_to_hidden1,
    'w_hidden1_to_hidden2': weights_hidden1_to_hidden2,
    'w_hidden2_to_output': weights_hidden2_to_output
}
biases = {
    'b_input_to_hidden1': bias_input_to_hidden1,
    'b_hidden1_to_hidden2': bias_hidden1_to_hidden2,
    'b_hidden_to_output': bias_hidden_to_output
}

# Сохранение обученной модели
save_model_npz(weights, biases)

# Загрузка модели обратно в память
loaded_model = load_model_npz()
print("Модель успешно загружена:")
print(loaded_model)


