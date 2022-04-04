from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Add, Flatten, Activation
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D


def resnet_layer(
        inputs,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation='relu',
        batch_normalization=True,
        conv_first=True
):
    conv = Conv2D(
        num_filters,
        kernel_size,
        strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4)
    )

    x = inputs

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        x = conv(x)

    return x


def resnet_v2(input_shape, depth, num_classes):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    num_filters_in = 16
    num_filters_out = 0
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape, name='input_image')

    x = resnet_layer(
        inputs,
        num_filters=num_filters_in,
    )

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False
            )

            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_in,
                conv_first=False
            )

            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False
            )

            if res_block == 0:
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation='',
                    batch_normalization=False
                )

            x = Add()([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='he_normal'
    )(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model
