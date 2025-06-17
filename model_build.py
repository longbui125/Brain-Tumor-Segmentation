from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore

def se_block(input_tensor, ratio=8):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = multiply([input_tensor, se])
    return x

def conv_block(inputs, num_filters, use_se=True, dropout_rate=0.2):
    x = Conv2D(num_filters, 3, padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if use_se:
        x = se_block(x, ratio=8)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def encoder_block(inputs, num_filters, use_se=True, dropout_rate=0.2):
    x = conv_block(inputs, num_filters, use_se, dropout_rate)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters, use_se=True, dropout_rate=0.2):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, use_se, dropout_rate)
    return x

def build_unet(input_shape=(256, 256, 3), dropout_rate=0.2):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64, dropout_rate=dropout_rate)
    s2, p2 = encoder_block(p1, 128, dropout_rate=dropout_rate)
    s3, p3 = encoder_block(p2, 256, dropout_rate=dropout_rate)
    s4, p4 = encoder_block(p3, 512, dropout_rate=dropout_rate)
    s5, p5 = encoder_block(p4, 1024, dropout_rate=dropout_rate)  # Extra depth

    b1 = conv_block(p5, 2048, dropout_rate=dropout_rate)

    d1 = decoder_block(b1, s5, 1024, dropout_rate=dropout_rate)
    d2 = decoder_block(d1, s4, 512, dropout_rate=dropout_rate)
    d3 = decoder_block(d2, s3, 256, dropout_rate=dropout_rate)
    d4 = decoder_block(d3, s2, 128, dropout_rate=dropout_rate)
    d5 = decoder_block(d4, s1, 64, dropout_rate=dropout_rate)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

    model = Model(inputs, outputs, name="CustomUNET")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()