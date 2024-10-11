import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image

# Update these paths according to your folder structure
image_dir = r'C:\Users\Dell\Downloads\voc_to_tfrecord\VOCdevkit\VOC2012\JPEGImages'
annotations_dir = r'C:\Users\Dell\Downloads\voc_to_tfrecord\VOCdevkit\VOC2012\Annotations'
output_path = r'C:\Users\Dell\Downloads\voc_to_tfrecord\output.tfrecord'

# Define class names for Pascal VOC
class_name_to_id = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

def create_tf_example(image_path, annotations):
    # Load image
    image = Image.open(image_path)
    width, height = image.size

    # Get bounding box information
    classes = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    for obj in annotations.findall('object'):
        class_name = obj.find('name').text
        if class_name in class_name_to_id:
            classes.append(class_name_to_id[class_name])
            bndbox = obj.find('bndbox')
            xmin.append(float(bndbox.find('xmin').text) / width)
            xmax.append(float(bndbox.find('xmax').text) / width)
            ymin.append(float(bndbox.find('ymin').text) / height)
            ymax.append(float(bndbox.find('ymax').text) / height)

    # Create TFRecord example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.gfile.GFile(image_path, 'rb').read()])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
    }))
    return tf_example

def main():
    writer = tf.io.TFRecordWriter(output_path)
    # List of common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    # Limit the number of images to process to 10
    image_files = image_files[:10]
    
    if not image_files:
        print("No images found in the directory.")
        return
    
    print("Found images (up to 10):")
    for image_file in image_files:
        print(image_file)  # Print image file name
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.join(annotations_dir, image_file.replace('.jpg', '.xml').replace('.jpeg', '.xml'))
        
        if os.path.exists(annotation_file):
            annotations = ET.parse(annotation_file).getroot()
            tf_example = create_tf_example(image_path, annotations)
            writer.write(tf_example.SerializeToString())
        else:
            print(f"No annotation file found for {image_file}")

    writer.close()
    print(f'TFRecord file "{output_path}" created successfully.')

if __name__ == '__main__':
    main()
