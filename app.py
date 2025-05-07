from flask import Flask, render_template, jsonify
import random
import numpy as np
import tensorflow as tf

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Daftar label untuk penyakit
class_labels = ['bronchitis', 'pertussis', 'asthma', 'pneumonia']

# Load model yang sudah dilatih
model = tf.keras.models.load_model('/Cough-Disease-Detection-using-CNN-Copy/Dataset/my_model.h5')

# Sample data X_test dan y_test
X_test = np.random.rand(20, 13, 100, 1)
y_test = np.random.randint(0, 4, 20)

# Route untuk halaman utama
@app.route('/')
def index():
    # Pilih data acak dari X_test dan y_test
    random_index = random.randint(0, len(X_test) - 1)
    input_data = X_test[random_index]
    input_label = y_test[random_index]
    
    # Reshape untuk model input
    input_data = input_data.reshape(1, *input_data.shape)
    
    # Prediksi menggunakan model
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)
    
    # Convert label dari one-hot ke index (jika perlu)
    actual_label_index = input_label
    
    return render_template('index.html', actual_label=class_labels[actual_label_index], 
                           predicted_label=class_labels[predicted_label], 
                           actual_label_index=actual_label_index, 
                           predicted_label_index=predicted_label)

# Route untuk mendapatkan prediksi baru tanpa reload halaman
@app.route('/predict')
def predict():
    random_index = random.randint(0, len(X_test) - 1)
    input_data = X_test[random_index]
    input_label = y_test[random_index]
    
    input_data = input_data.reshape(1, *input_data.shape)
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)
    
    actual_label_index = input_label
    
    return jsonify({
        "actual": class_labels[actual_label_index],
        "predicted": class_labels[predicted_label]
    })

if __name__ == "__main__":
    app.run(debug=True)
