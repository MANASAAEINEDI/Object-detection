import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Path to your TFRecord file
tfrecord_path = r'C:\Users\Dell\Downloads\voc_to_tfrecord\output.tfrecord'

# Define class names for Pascal VOC
class_id_to_name = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

# Function to parse the TFRecord file
def _parse_function(proto):
    # Define your tfrecord structure
    keys_to_features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    }
    # Parse the input tf.train.Example proto using the dictionary above
    return tf.io.parse_single_example(proto, keys_to_features)

# Read the TFRecord file
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

# Initialize lists for true and predicted labels
true_labels = []
predicted_labels = []

# Set maximum number of images to display
max_images_to_display = 10
image_count = 0

# Iterate through the dataset
for raw_record in raw_dataset:
    if image_count >= max_images_to_display:
        break  # Stop after displaying the desired number of images

    example = _parse_function(raw_record)

    # Get the actual filename
    filename = example['image/filename'].numpy().decode()

    # Print image details
    print(f"Image Filename: {filename}")
    print("Image Height:", example['image/height'].numpy())
    print("Image Width:", example['image/width'].numpy())
    print("Encoded Image Data Length:", len(example['image/encoded'].numpy()))

    # Get class labels and map them to names
    class_labels = example['image/object/class/label'].values.numpy()
    class_names = [class_id_to_name[label] for label in class_labels]

    # Collect true labels
    true_labels.extend(class_labels)

    # Simulate predictions (here we just use the first label for demonstration)
    predicted_labels.extend([class_labels[0]] * len(class_labels))

    # Print object class labels
    print("Object Class Labels:", class_names)

    # Print bounding boxes
    print("Bounding Boxes (xmin, xmax, ymin, ymax):")
    print("Xmin:", example['image/object/bbox/xmin'].values.numpy())
    print("Xmax:", example['image/object/bbox/xmax'].values.numpy())
    print("Ymin:", example['image/object/bbox/ymin'].values.numpy())
    print("Ymax:", example['image/object/bbox/ymax'].values.numpy())

    # Display the image
    image_data = example['image/encoded'].numpy()
    image = tf.image.decode_jpeg(image_data, channels=3)  # Assuming JPEG format
    plt.imshow(image.numpy())
    plt.title(f"{filename}: {', '.join(class_names)}")
    plt.axis('off')
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Pause to allow the image to render
    print("\n")

    image_count += 1  # Increment image count

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
