
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

LATENT_DIM   = 128          # size of noise vector z
IMG_SHAPE    = (32, 32, 3)
TARGET_CLASS = 5             # 5 = "dog" in CIFAR-10
CLASS_NAMES  = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
print(f"Target class: {TARGET_CLASS} → {CLASS_NAMES[TARGET_CLASS]}")

(x_train_full, y_train_full), (x_test_full, y_test_full) = (
    keras.datasets.cifar10.load_data()
)

mask_train = (y_train_full.squeeze() == TARGET_CLASS)
mask_test  = (y_test_full.squeeze()  == TARGET_CLASS)

x_train = x_train_full[mask_train].astype("float32")
x_test  = x_test_full[mask_test].astype("float32")

x_train = (x_train - 127.5) / 127.5
x_test  = (x_test  - 127.5) / 127.5

print(f"Training images ({CLASS_NAMES[TARGET_CLASS]}): {x_train.shape[0]}")
print(f"Test images     ({CLASS_NAMES[TARGET_CLASS]}): {x_test.shape[0]}")

fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow((x_train[i] * 0.5 + 0.5))  # rescale back to [0,1]
    ax.axis("off")
fig.suptitle(f"Real {CLASS_NAMES[TARGET_CLASS]} images", fontsize=13)
plt.tight_layout()
plt.show()


def build_generator(latent_dim=LATENT_DIM):
    z = keras.Input(shape=(latent_dim,), name="z_input")

    # project & reshape: 128 → 4×4×256
    x = layers.Dense(4 * 4 * 256, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 256))(x)          # (4, 4, 256)

    # up-conv block 1: 4×4 → 8×8
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                         # (8, 8, 128)

    # up-conv block 2: 8×8 → 16×16
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                         # (16, 16, 64)

    # up-conv block 3: 16×16 → 32×32
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False,
                               activation="tanh")(x)  # (32, 32, 3)

    model = keras.Model(z, x, name="generator")
    return model

generator = build_generator()
generator.summary()


def build_discriminator():
    img = keras.Input(shape=IMG_SHAPE, name="image")

    # conv block 1: 32×32 → 16×16
    x = layers.Conv2D(64, 4, strides=2, padding="same")(img)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    # conv block 2: 16×16 → 8×8
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    # conv block 3: 8×8 → 4×4
    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)     # logit

    model = keras.Model(img, x, name="discriminator")
    return model

discriminator = build_discriminator()
discriminator.summary()


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss(real_out, fake_out):
    return (cross_entropy(tf.ones_like(real_out),  real_out) +
            cross_entropy(tf.zeros_like(fake_out), fake_out))

def gen_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)

gen_opt  = keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(real_images):
    batch = tf.shape(real_images)[0]
    noise = tf.random.normal([batch, LATENT_DIM])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_images = generator(noise, training=True)

        real_out = discriminator(real_images, training=True)
        fake_out = discriminator(fake_images, training=True)

        g_loss = gen_loss(fake_out)
        d_loss = disc_loss(real_out, fake_out)

    g_grads = gt.gradient(g_loss, generator.trainable_variables)
    d_grads = dt.gradient(d_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss, d_loss

fixed_seed = tf.random.normal([16, LATENT_DIM], seed=SEED)

def show_generated(epoch, seed=fixed_seed):
    imgs = generator(seed, training=False).numpy()
    imgs = imgs * 0.5 + 0.5            # [-1,1] → [0,1]
    imgs = np.clip(imgs, 0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        ax.axis("off")
    fig.suptitle(f"Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.show()

BATCH_SIZE = 128
EPOCHS     = 80      # adjust down if very slow on CPU

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(x_train.shape[0], seed=SEED)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

g_losses, d_losses = [], []

for epoch in range(1, EPOCHS + 1):
    for batch in dataset:
        gl, dl = train_step(batch)
    g_losses.append(float(gl))
    d_losses.append(float(dl))

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  |  G loss {gl:.4f}  |  D loss {dl:.4f}")
        show_generated(epoch)


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(g_losses, label="Generator loss")
ax.plot(d_losses, label="Discriminator loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("DCGAN Training Losses")
ax.legend()
plt.tight_layout()
plt.show()


noise = tf.random.normal([32, LATENT_DIM])
final_imgs = generator(noise, training=False).numpy() * 0.5 + 0.5
final_imgs = np.clip(final_imgs, 0, 1)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(final_imgs[i])
    ax.axis("off")
fig.suptitle(f"Final Generated {CLASS_NAMES[TARGET_CLASS]} Images", fontsize=14)
plt.tight_layout()
plt.show()


z1 = tf.random.normal([1, LATENT_DIM], seed=0)
z2 = tf.random.normal([1, LATENT_DIM], seed=99)

n_steps = 10
alphas = np.linspace(0, 1, n_steps)
interp_z = np.array([(1 - a) * z1.numpy() + a * z2.numpy() for a in alphas])
interp_z = interp_z.reshape(n_steps, LATENT_DIM)

interp_imgs = generator(interp_z, training=False).numpy() * 0.5 + 0.5
interp_imgs = np.clip(interp_imgs, 0, 1)

fig, axes = plt.subplots(1, n_steps, figsize=(18, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(interp_imgs[i])
    ax.set_title(f"α={alphas[i]:.1f}", fontsize=9)
    ax.axis("off")
fig.suptitle("Latent-Space Interpolation (z₁ → z₂)", fontsize=13)
plt.tight_layout()
plt.show()


base_z = tf.random.normal([1, LATENT_DIM], seed=7).numpy()
dim_to_vary = 0          # try different dims: 0, 10, 50, 100 …
vals = np.linspace(-3, 3, 10)

sweep_z = np.tile(base_z, (len(vals), 1))
sweep_z[:, dim_to_vary] = vals

sweep_imgs = generator(sweep_z, training=False).numpy() * 0.5 + 0.5
sweep_imgs = np.clip(sweep_imgs, 0, 1)

fig, axes = plt.subplots(1, len(vals), figsize=(18, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(sweep_imgs[i])
    ax.set_title(f"z[{dim_to_vary}]={vals[i]:.1f}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Varying dimension {dim_to_vary} of latent vector", fontsize=13)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 8, figsize=(14, 4))

z_normal    = tf.random.normal([8, LATENT_DIM], seed=42)
z_truncated = tf.random.truncated_normal([8, LATENT_DIM], stddev=0.5, seed=42)

for i in range(8):
    img_n = generator(z_normal[i:i+1], training=False).numpy()[0]*0.5+0.5
    img_t = generator(z_truncated[i:i+1], training=False).numpy()[0]*0.5+0.5
    axes[0, i].imshow(np.clip(img_n, 0, 1))
    axes[0, i].axis("off")
    axes[1, i].imshow(np.clip(img_t, 0, 1))
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Normal z", fontsize=11)
axes[1, 0].set_ylabel("Truncated z", fontsize=11)
fig.suptitle("Standard vs Truncated Noise Sampling", fontsize=13)
plt.tight_layout()
plt.show()


za = tf.random.normal([1, LATENT_DIM], seed=1)
zb = tf.random.normal([1, LATENT_DIM], seed=2)
zc = tf.random.normal([1, LATENT_DIM], seed=3)
zd = za - zb + zc            # arithmetic result

labels = ["z_a", "z_b", "z_c", "z_a − z_b + z_c"]
vectors = [za, zb, zc, zd]

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, (lbl, zv) in enumerate(zip(labels, vectors)):
    img = generator(zv, training=False).numpy()[0]*0.5+0.5
    axes[i].imshow(np.clip(img, 0, 1))
    axes[i].set_title(lbl, fontsize=10)
    axes[i].axis("off")
fig.suptitle("Latent-Space Arithmetic", fontsize=13)
plt.tight_layout()
plt.show()


