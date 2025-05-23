import numpy as np
try:
    import onnxruntime as rt
except ImportError:
    rt = None

import cv2

from duckiebots_unreal_sim.tools.image_renderer import ImageRenderer


have_tensorflow = False
try:
    import tensorflow as tf
    have_tensorflow = True
except ImportError:
    pass

def onehot_to_rgb(onehot):
    color_dict = {0: [0, 0, 0], 1: [251, 244, 4], 2: [255, 255, 255], 3: [255, 0, 255]}
    # black, yellow, white, pink

    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:-1] + (3,), dtype=np.uint8)
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return output


if have_tensorflow:

    # Set up device
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.LayerNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.LayerNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result


    def GeneratorSmall(output_channels=4):
        # print(vars.IMG_WIDTH)
        inputs = tf.keras.layers.Input(shape=[64, 64, 3])

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation=None)  # (batch_size, 256, 256, 4) # activation was 'tanh' for earlier experiments

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        # for skip in skips:
        #     print(skip)
        # print("skip last", skips[-1])
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            # print(up.summary())
            # print("concatenate", "up", x, "down", skip)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def normalize(input_image):
        input_image = (tf.cast(input_image, dtype=tf.float32) / 127.5) - 1
        return input_image


    class RCANObsPreprocessor:
        def __init__(self, checkpoint_path: str, debug_render_predictions: bool = False):
            self.generator = GeneratorSmall()
            self.generator.summary()
            # Load the checkpoint.
            checkpoint = tf.train.Checkpoint(generator=self.generator)
            checkpoint.restore(checkpoint_path)

            self._debug_render_predictions = debug_render_predictions
            if self._debug_render_predictions:
                self._image_renderer = ImageRenderer()
                self._image_renderer2 = ImageRenderer()

        def preprocess_obs(self, bgr_obs: np.ndarray) -> np.ndarray:
            assert bgr_obs.shape == (60, 80, 3), bgr_obs.shape

            if self._debug_render_predictions:
                self._image_renderer2.render_cv2_image(cv2_image=bgr_obs)

            rgb_image = np.asarray(bgr_obs, dtype=np.uint8)[::-1, ::, ::-1]
            rgb_image = tf.image.pad_to_bounding_box(
                image=[rgb_image],
                offset_height=0,
                offset_width=0,
                target_height=80,
                target_width=80
            )
            rgb_image = tf.image.resize(rgb_image,
                                        size=[64, 64],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            rgb_image = normalize(rgb_image)

            raw_prediction = self.generator(rgb_image, training=False)  # todo: Why set to True?
            pred_resized = tf.image.resize(raw_prediction,
                                           size=[80, 80],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            pred_resized = pred_resized[:, :60, :, :][0]
            pred_resized = onehot_to_rgb(pred_resized)[::-1, :, ::-1]

            if self._debug_render_predictions:
                self._image_renderer.render_cv2_image(cv2_image=pred_resized)
            assert pred_resized.shape == (60, 80, 3), pred_resized.shape
            return pred_resized

class ONNXRCANObsPreprocessor:
    def __init__(self, checkpoint_path: str, debug_render_predictions: bool = False):
        # providers = ['CPUExecutionProvider']
        providers = ['CUDAExecutionProvider']
        if rt is None:
            raise ImportError("Couldn't import onnx runtime.")

        self._onnx_model = rt.InferenceSession(path_or_bytes=checkpoint_path, providers=providers)

        self._debug_render_predictions = debug_render_predictions
        if self._debug_render_predictions:
            self._image_renderer = ImageRenderer()
            self._image_renderer2 = ImageRenderer()

    def preprocess_obs(self, rgb_obs: np.ndarray) -> np.ndarray:
        # assert rgb_obs.shape == (60, 80, 3), rgb_obs.shape
        # rgb_image = np.asarray(rgb_obs, dtype=np.uint8)
        # padded_rgb_image = np.zeros(shape=(80, 80, 3), dtype=np.uint8)
        # padded_rgb_image[:60, :, :] = rgb_image
        # padded_rgb_image = cv2.resize(src=padded_rgb_image, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        # input_image = (np.asarray(padded_rgb_image, dtype=np.float32) / 127.5) - 1
        rgb_obs = np.asarray(rgb_obs, dtype=np.float32)
        if len(rgb_obs.shape) == 3:
            rgb_obs = cv2.resize(src=rgb_obs, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
            input_image = (rgb_obs / 127.5) - 1
            onnx_pred = self._onnx_model.run(['conv2d_transpose_5'], {"input": [input_image]})[0][0]
            rgb_pred = onehot_to_rgb(onnx_pred)
            if self._debug_render_predictions:
                self._image_renderer.render_cv2_image(cv2_image=rgb_pred)
            assert rgb_pred.shape == (64, 64, 3), rgb_pred.shape
        else:
            assert len(rgb_obs.shape) == 4, rgb_obs.shape
            rgb_obs = np.asarray([cv2.resize(src=im, dsize=(64, 64), interpolation=cv2.INTER_NEAREST) for im in rgb_obs], dtype=np.float32)
            input_image = (rgb_obs / 127.5) - 1
            onnx_pred = self._onnx_model.run(['conv2d_transpose_5'], {"input": input_image})[0]
            rgb_pred = onehot_to_rgb(onnx_pred)
            assert rgb_pred.shape == (rgb_obs.shape[0], 64, 64, 3), rgb_pred.shape

        # if self._debug_render_predictions:
        #
        #     self._image_renderer2.render_cv2_image(cv2_image=input_image)


        # if self._debug_render_predictions:
        #     print(input_image)
        #     print(input_image.dtype)
        #     self._image_renderer.render_cv2_image(cv2_image=onnx_pred)

        # pred_resized = cv2.resize(src=onnx_pred, dsize=(80, 80), interpolation=cv2.INTER_NEAREST)
        # pred_resized = pred_resized[:60, :, :]
        # pred_resized = onehot_to_rgb(pred_resized)
        #
        # if self._debug_render_predictions:
        #     self._image_renderer.render_cv2_image(cv2_image=pred_resized)
        # assert pred_resized.shape == (60, 80, 3), pred_resized.shape
        # return pred_resized

        return rgb_pred


if __name__ == '__main__':
    from duckiebots_unreal_sim.holodeck_lane_following_env import UEDuckiebotsHolodeckEnv

    env_config = {
        "use_domain_randomization": True,
        "render_game_on_screen": False,
        "use_mask_observation": False,
        "return_rgb_and_mask_as_observation": False,
        "simulate_latency": False
    }

    env = UEDuckiebotsHolodeckEnv(
        randomization_enabled=env_config["use_domain_randomization"],
        physics_hz=20. if env_config["simulate_latency"] else 10.,
        physics_ticks_between_action_and_observation=1,
        physics_ticks_between_observation_and_action=1 if env_config["simulate_latency"] else 0,
        render_game_on_screen=env_config["render_game_on_screen"],
        return_only_mask_as_observation=env_config["use_mask_observation"],
        return_rgb_and_mask_as_observation=env_config["return_rgb_and_mask_as_observation"],
        # reward_function=env_config["reward_function"],
        randomize_mask=True,
    )

    rcan2 = ONNXRCANObsPreprocessor(debug_render_predictions=True)

    while True:
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            assert obs.shape == (60, 80, 3), obs.shape
            # env.render(image_scale=10)
            # print(rcan.preprocess_obs(bgr_obs=obs))
            rcan2.preprocess_obs(bgr_obs=obs)
