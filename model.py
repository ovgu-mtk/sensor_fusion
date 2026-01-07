import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class TemporalAttention(layers.Layer):
    """
    learns which timestep in sequence is more important.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = layers.Dense(self.units, name='attention_W')
        self.V = layers.Dense(1, name='attention_V')
        super().build(input_shape)

    def call(self, x):
        # x: (batch, seq_len, features)
        score = self.V(tf.nn.tanh(self.W(x)))  # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * x, axis=1)  # (batch, features)
        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ResidualDKF(layers.Layer):
    def __init__(self, latent_dim=64, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def build(self, input_shape):

        self.transition = layers.GRU(self.latent_dim, return_sequences=True)
        self.delta_mu = layers.Dense(self.latent_dim)
        self.state_logvar = layers.Dense(self.latent_dim)
        self.x_proj = layers.Dense(self.latent_dim)

        # Observation Decoder
        self.obs_mu = layers.Dense(2, name="obs_mu")
        self.obs_logvar = layers.Dense(2, name="obs_logvar")

        # Velocity Decoder
        self.velocity_mu = layers.Dense(2, name="velocity_mu")
        self.velocity_logvar = layers.Dense(2, name="velocity_logvar")

        super().build(input_shape)

    def call(self, x, training=None):
        h = self.transition(x)

        delta_mu = self.delta_mu(h)
        logvar_z = self.state_logvar(h)

        if training:
            std_z = tf.exp(0.5 * logvar_z)
            delta_z = delta_mu + std_z * tf.random.normal(tf.shape(delta_mu))

            kl_loss = -0.5 * tf.reduce_mean(
                1 + logvar_z - tf.square(delta_mu) - tf.exp(logvar_z)
            )
            self.add_loss(self.kl_weight * kl_loss)
        else:
            delta_z = delta_mu

        z = self.x_proj(x) + delta_z

        # Position Reconstruction
        obs_mu = self.obs_mu(z)
        obs_logvar = self.obs_logvar(z)

        # Velocity Estimation (from latent state)
        vel_mu = self.velocity_mu(z)
        vel_logvar = self.velocity_logvar(z)

        # Output: [z, pos_mu, pos_logvar, vel_mu, vel_logvar]
        return tf.concat([z, obs_mu, obs_logvar, vel_mu, vel_logvar], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
        })
        return config


class AdaptiveSensorFusionDKF(layers.Layer):
    """
    Deep Kalman Filter with adaptive weighting.
    learns which sensor is more robust at the moment.
    """

    def __init__(self, latent_dim=64, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def build(self, input_shape):
        # ====================================
        # SENSOR UNCERTAINTY ESTIMATION
        # ====================================
        # each sensor get it own network for confidence
        self.uwb_confidence = keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 0-1 Confidence Score
        ], name='uwb_confidence')

        self.grid_confidence = keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='grid_confidence')

        # ====================================
        # ADAPTIVE GATING
        # ====================================
        self.sensor_gate = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')  # [weight_uwb, weight_grid]
        ], name='sensor_gate')

        # ====================================
        # SENSOR-SPECIFIC ENCODERS
        # ====================================
        # each sensor get own position in latent spaace
        self.uwb_encoder = layers.Dense(self.latent_dim, name='uwb_encoder')
        self.grid_encoder = layers.Dense(self.latent_dim, name='grid_encoder')

        # ====================================
        # KALMAN FILTER CORE
        # ====================================
        # Transition Model (learns dynamic)
        self.transition = layers.GRU(self.latent_dim, return_sequences=True)

        # Variational Parameters
        self.delta_mu = layers.Dense(self.latent_dim, name="delta_mu")
        self.state_logvar = layers.Dense(self.latent_dim, name="state_logvar")

        # ====================================
        # DECODERS
        # ====================================
        self.obs_mu = layers.Dense(2, name="obs_mu")
        self.obs_logvar = layers.Dense(2, name="obs_logvar")
        self.velocity_mu = layers.Dense(2, name="velocity_mu")
        self.velocity_logvar = layers.Dense(2, name="velocity_logvar")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        inputs: [uwb_input, grid_features]
        - uwb_input: (batch, seq_len, 2)
        - grid_features: (batch, seq_len, 32)
        """
        uwb_input, grid_features = inputs

        # ====================================
        # 1. SENSOR CONFIDENCE ESTIMATION
        # ====================================
        uwb_conf = self.uwb_confidence(uwb_input)  # (batch, seq, 1)
        grid_conf = self.grid_confidence(grid_features)  # (batch, seq, 1)

        # ====================================
        # 2. ADAPTIVE SENSOR WEIGHTING
        # ====================================
        sensor_concat = tf.concat([uwb_input, grid_features], axis=-1)
        sensor_weights = self.sensor_gate(sensor_concat)  # (batch, seq, 2)

        # extracts weights
        w_uwb = sensor_weights[..., 0:1]  # (batch, seq, 1)
        w_grid = sensor_weights[..., 1:2]  # (batch, seq, 1)

        # multiply with confidence
        w_uwb = w_uwb * uwb_conf
        w_grid = w_grid * grid_conf

        # normalize
        total_weight = w_uwb + w_grid + 1e-8
        w_uwb = w_uwb / total_weight
        w_grid = w_grid / total_weight

        # ====================================
        # 3. WEIGHTED SENSOR FUSION
        # ====================================
        uwb_encoded = self.uwb_encoder(uwb_input)
        grid_encoded = self.grid_encoder(grid_features)

        # weighted fusion
        fused_features = w_uwb * uwb_encoded + w_grid * grid_encoded

        # ====================================
        # 4. KALMAN FILTER
        # ====================================
        h = self.transition(fused_features)

        delta_mu = self.delta_mu(h)
        logvar_z = self.state_logvar(h)

        if training:
            std_z = tf.exp(0.5 * logvar_z)
            eps = tf.random.normal(tf.shape(delta_mu))
            delta_z = delta_mu + std_z * eps

            kl_loss = -0.5 * tf.reduce_mean(
                1 + logvar_z - tf.square(delta_mu) - tf.exp(logvar_z)
            )
            self.add_loss(self.kl_weight * kl_loss)
        else:
            delta_z = delta_mu

        z = fused_features + delta_z

        # ====================================
        # 5. OUTPUTS
        # ====================================
        obs_mu = self.obs_mu(z)
        obs_logvar = self.obs_logvar(z)
        vel_mu = self.velocity_mu(z)
        vel_logvar = self.velocity_logvar(z)

        return tf.concat([z, obs_mu, obs_logvar, vel_mu, vel_logvar], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
        })
        return config

# ========================================
# Modelle
# ========================================


def create_minimal_multimodal_model(
        grid_size=500,
        num_layers=1,
        sequence_length=10,
        gru_units=64,
):
    """
    Absolute minimal viable architecture.
    Used for baseline comparison.
    """

    grid_input = layers.Input(
        shape=(grid_size, grid_size, num_layers, sequence_length),
        name='grid_input'
    )
    uwb_input = layers.Input(
        shape=(sequence_length, 2),
        name='uwb_input'
    )

    # ========================================
    # GRID: Minimal CNN
    # ========================================
    x_grid = layers.Permute((4, 1, 2, 3))(grid_input)
    x_grid = layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), strides=(4, 4), padding='same')
    )(x_grid)
    x_grid = layers.TimeDistributed(layers.ReLU())(x_grid)
    x_grid = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x_grid)

    # ========================================
    # UWB: Simple GRU (no DKF)
    # ========================================
    x_uwb = layers.GRU(gru_units, return_sequences=True)(uwb_input)

    # ========================================
    # FUSION: Concatenate + GRU
    # ========================================
    fused = layers.Concatenate()([x_uwb, x_grid])
    x = layers.GRU(gru_units, return_sequences=False)(fused)

    # ========================================
    # OUTPUT
    # ========================================
    x = layers.Dense(32, activation='relu')(x)
    position_output = layers.Dense(2, activation='linear', name='position')(x)

    outputs = {'position': position_output}

    model = keras.Model(
        inputs={'grid_input': grid_input, 'uwb_input': uwb_input},
        outputs=outputs,
        name='minimal_multimodal_model'
    )

    return model




def create_kalman_multimodal_model(
        grid_size=500,
        num_layers=1,
        sequence_length=10,
        latent_dim=64,
        use_velocity_auxiliary=True,
        gru_units=96,
):

    # ========================================
    # INPUTS
    # ========================================
    grid_input = layers.Input(
        shape=(grid_size, grid_size, num_layers, sequence_length),
        name='grid_input'
    )
    uwb_input = layers.Input(
        shape=(sequence_length, 2),
        name='uwb_input'
    )

    # ========================================
    # GRID: Minimal CNN
    # ========================================
    x_grid = layers.Permute((4, 1, 2, 3))(grid_input)
    x_grid = layers.TimeDistributed(layers.Conv2D(32, (5, 5), strides=(4, 4), padding='same'))(x_grid)
    x_grid = layers.TimeDistributed(layers.ReLU())(x_grid)
    x_grid = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x_grid)

    # ========================================
    # UWB PROCESSING
    # ========================================
    dkf_output = ResidualDKF(latent_dim=latent_dim, name='dkf_velocity')(uwb_input)

    # Simple encoding (no separate extraction needed)
    x_uwb = layers.Dense(latent_dim, activation='relu', name='uwb_features')(dkf_output)
    x_uwb = layers.LayerNormalization()(x_uwb)

    # ========================================
    # SIMPLE FUSION
    # ========================================
    # Just concatenate both modalities
    fused = layers.Concatenate(name='fusion')([x_uwb, x_grid])

    # Single dense layer for fusion
    fused = layers.Dense(latent_dim * 2, activation='relu')(fused)
    fused = layers.Dropout(0.15)(fused)

    # ========================================
    # TEMPORAL MODELING
    # ========================================
    x_temporal = layers.Bidirectional(layers.GRU(gru_units, dropout=0.2, return_sequences=True), name='bi_gru')(fused)

    # Simple temporal attention (aggregate sequence)
    context, attention_weights = TemporalAttention(units=128, name='temp_attn')(x_temporal)

    # ========================================
    # OUTPUT HEADS - Streamlined
    # ========================================
    # Shared decoder
    x_out = layers.Dense(64, activation='relu', name='decoder')(context)
    x_out = layers.Dropout(0.15)(x_out)

    # Position output (main task)
    position_output = layers.Dense(2, activation='linear', name='position')(x_out)

    outputs = {'position': position_output}

    # Velocity auxiliary (optional)
    if use_velocity_auxiliary:
        outputs['velocity_aux'] = layers.Lambda(lambda x: x[:, -1, latent_dim + 4:latent_dim + 6], name='velocity_aux')(dkf_output)

    # ========================================
    # MODEL
    # ========================================
    model = keras.Model(
        inputs={'grid_input': grid_input, 'uwb_input': uwb_input},
        outputs=outputs,
        name='kalman_multimodal_model'
    )

    return model


def create_fused_kalman_multimodal_model(
        grid_size=500,
        num_layers=1,
        sequence_length=10,
        latent_dim=64,
        gru_units=96,
        use_velocity_auxiliary=True,
):
    # --- INPUTS ---
    grid_input = layers.Input(shape=(grid_size, grid_size, num_layers, sequence_length), name='grid_input')
    uwb_input = layers.Input(shape=(sequence_length, 2), name='uwb_input')  # (x, y)

    # ========================================
    # GRID: Minimal CNN
    # ========================================
    x_grid = layers.Permute((4, 1, 2, 3))(grid_input)
    x_grid = layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), strides=(4, 4), padding='same')
    )(x_grid)
    x_grid = layers.TimeDistributed(layers.ReLU())(x_grid)
    x_grid = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x_grid)
    grid_features = layers.TimeDistributed(layers.Dense(32, activation='relu'), name='grid_embedding')(x_grid)

    # ========================================
    # EARLY FUSION --- UWB (2 Features) + Grid (32 Features)
    # ========================================
    fused_sensor_input = layers.Concatenate(axis=-1, name='sensor_fusion')([uwb_input, grid_features])

    # KALMAN FILTER
    dkf_output = ResidualDKF(latent_dim=latent_dim, kl_weight=0.01, name='fused_dkf')(fused_sensor_input)

    # ========================================
    # TEMPORAL MODELING
    # ========================================
    x_temporal = layers.Bidirectional(layers.GRU(gru_units, dropout=0.2, return_sequences=True), name='bi_gru')(dkf_output)

    # Simple temporal attention (aggregate sequence)
    context, attention_weights = TemporalAttention(units=128, name='temp_attn')(x_temporal)

    # ========================================
    # OUTPUT HEADS
    # ========================================
    x_out = layers.Dense(64, activation='relu', name='decoder')(context)
    x_out = layers.Dropout(0.15)(x_out)

    # Position output
    position_output = layers.Dense(2, activation='linear', name='position')(x_out)

    outputs = {'position': position_output}

    # Velocity auxiliary (optional)
    if use_velocity_auxiliary:
        outputs['velocity_aux'] = layers.Lambda(lambda x: x[:, -1, latent_dim + 4:latent_dim + 6], name='velocity_aux')(
            dkf_output)

    model = keras.Model(
        inputs=[grid_input, uwb_input],
        outputs=outputs,
        name="fused_kalman_multimodal_model"
    )
    return model



def create_adaptive_fused_model(
        grid_size=500,
        num_layers=1,
        sequence_length=10,
        latent_dim=64,
        gru_units=96,
        use_velocity_auxiliary=True):

    # Inputs
    grid_input = layers.Input(shape=(grid_size, grid_size, num_layers, sequence_length), name='grid_input')
    uwb_input = layers.Input(shape=(sequence_length, 2), name='uwb_input')

    # ========================================
    # GRID: Minimal CNN
    # ========================================
    x_grid = layers.Permute((4, 1, 2, 3))(grid_input)
    x_grid = layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), strides=(4, 4), padding='same')
    )(x_grid)
    x_grid = layers.TimeDistributed(layers.ReLU())(x_grid)
    x_grid = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x_grid)

    # ========================================
    # FUSION --- ADAPTIVE KALMAN FILTER
    # ========================================
    dkf_output = AdaptiveSensorFusionDKF(latent_dim=latent_dim, kl_weight=0.01, name='adaptive_dkf')([uwb_input, x_grid])

    # ========================================
    # TEMPORAL MODELING
    # ========================================
    x_temporal = layers.Bidirectional(layers.GRU(gru_units, dropout=0.2, return_sequences=True), name='bi_gru')(
        dkf_output)

    # Simple temporal attention (aggregate sequence)
    context, attention_weights = TemporalAttention(units=128, name='temp_attn')(x_temporal)

    # ========================================
    # OUTPUT HEADS
    # ========================================
    x_out = layers.Dense(64, activation='relu', name='decoder')(context)
    x_out = layers.Dropout(0.15)(x_out)

    # Position output
    position_output = layers.Dense(2, activation='linear', name='position')(x_out)

    outputs = {'position': position_output}

    # Velocity auxiliary (optional)
    if use_velocity_auxiliary:
        outputs['velocity_aux'] = layers.Lambda(lambda x: x[:, -1, latent_dim + 4:latent_dim + 6], name='velocity_aux')(
            dkf_output)

    model = keras.Model(
        inputs=[grid_input, uwb_input],
        outputs=outputs,
        name="adaptive_fused_model"
    )
    return model


# ========================================
# TRAINING SETUP
# ========================================

class TrainingSetup:

    @staticmethod
    def compile_model(model, loss='velocity_auxiliary', learning_rate=1e-4):

        if loss == 'velocity_auxiliary':
            model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=1e-5
                ),
                loss={
                    'position': 'huber',  # More robust than MSE
                    'velocity_aux': 'mse'
                },
                loss_weights={
                    'position': 1.0,
                    'velocity_aux': 0.2  # Reduced from 0.3 - auxiliary should be subtle
                },
                metrics={
                    'position': [
                        'mae',
                        keras.metrics.RootMeanSquaredError(name='rmse')
                    ],
                    'velocity_aux': ['mae']
                }
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=1e-5),
                loss={
                    'position': 'huber'},
                metrics={
                    'position': [
                        'mae',
                        keras.metrics.RootMeanSquaredError(name='rmse')
                    ]
                }
            )

        return model

    @staticmethod
    def get_callbacks(save_model_path, patience=6, monitor='val_loss'):
        """
        Standard Callbacks für Training.
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                save_model_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]

        return callbacks