import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 정의
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=10, batch_siz = 32, validation_split=0.2)

# 모델 평가
model.evaluate(x_test, y_test, verbose=2)

# 예측 결과 출력
results = model.predict(x_test[:10])
print(tf.argmax(results, axis=1))
print(y_test[:10])