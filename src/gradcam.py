
import numpy as np
import tensorflow as tf


LAST_CONV_LAYER = "out_relu"


def build_gradcam_model(model, last_conv_layer_name=LAST_CONV_LAYER):
    """
    Build a Grad-CAM model supporting direct and nested
    MobileNetV2 architectures.
    """

    # COVID: MobileNetV2 is directly inside the model.
    try:
        backbone = model.get_layer(last_conv_layer_name)
        raise ValueError("Unexpected layer match")
    except ValueError:
        try:
            backbone = model
            conv_layer = backbone.get_layer(last_conv_layer_name)
            backbone_index = -1
        except ValueError:
            backbone = None

    # Other four datasets: MobileNetV2 is nested.
    if backbone is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    layer.get_layer(last_conv_layer_name)
                    backbone = layer
                    break
                except ValueError:
                    continue

    if backbone is None:
        raise ValueError(
            f"Could not find '{last_conv_layer_name}' "
            "in the model or its nested backbones."
        )

    conv_layer = backbone.get_layer(last_conv_layer_name)

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=conv_layer.output,
        name="gradcam_feature_extractor"
    )

    # Reconstruct the classifier using the original trained layers.
    if backbone is model:
        start_index = model.layers.index(conv_layer) + 1
    else:
        start_index = model.layers.index(backbone) + 1

    classifier_layers = model.layers[start_index:]

    if not classifier_layers:
        raise ValueError("No classifier layers found after backbone.")

    feature_input = tf.keras.Input(
        shape=tuple(feature_extractor.output.shape[1:]),
        name="gradcam_feature_input"
    )

    x = feature_input

    for layer in classifier_layers:
        x = layer(x, training=False)

    classifier = tf.keras.Model(
        inputs=feature_input,
        outputs=x,
        name="gradcam_classifier"
    )

    return feature_extractor, classifier


def make_gradcam_heatmap(
    image,
    model,
    predicted_class_index=None,
    last_conv_layer_name=LAST_CONV_LAYER
):
    """Generate a class-specific Grad-CAM heatmap."""

    feature_extractor, classifier = build_gradcam_model(
        model,
        last_conv_layer_name
    )

    # Extract feature maps.
    feature_maps = feature_extractor(
        image,
        training=False
    )

    with tf.GradientTape() as tape:

        # Watch feature maps to calculate class-score gradients.
        tape.watch(feature_maps)

        predictions = classifier(
            feature_maps,
            training=False
        )

        if predicted_class_index is None:
            predicted_class_index = int(
                tf.argmax(predictions[0]).numpy()
            )

        class_score = predictions[
            :, predicted_class_index
        ]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Global average pooling of gradients.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Weighted combination of feature maps.
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        maximum + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def resize_heatmap(
    heatmap,
    target_size=(224, 224)
):
    """Resize Grad-CAM heatmap to image dimensions."""

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        target_size
    )

    return tf.squeeze(heatmap).numpy()
