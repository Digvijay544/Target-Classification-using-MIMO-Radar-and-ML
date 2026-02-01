from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv3D(8, (3,3,3), activation='relu'),
        MaxPooling3D((2,2,2)),

        Conv3D(16, (3,3,3), activation='relu'),
        MaxPooling3D((2,2,2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
