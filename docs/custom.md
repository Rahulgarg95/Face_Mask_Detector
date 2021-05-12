# Custom Model Training
----
Training Models using images can be quite tricky, hence below steps will guide to make process easy and will save lot of time.


## Download required files/tools
----

* [Download](https://github.com/tensorflow/models/tree/v1.13.0) v1.13.0 model.
* [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) the faster_rcnn_inception_v2_coco model from the model zoo **or** any other model of your choice from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md" target="_blank">TensorFlow 1 Detection Model Zoo.</a>
* [Download](https://drive.google.com/file/d/166BzBdrYAL0naFMGKQbTtQV9B_lKcjph/view?usp=sharing) Mask Dataset from Google Drive.
* [Download](https://tzutalin.github.io/labelImg/) labelImg tool for labeling images.


## Installation Steps (Google Colab)
----

#### *Step - 1: Create a new notebook with Google Colab.*

    1. Visit Colab Website -> https://colab.research.google.com/notebooks/intro.ipynb
    2. File > New notebook

#### *Step - 2: Change runtime to GPU*

    Runtime > Change runtime type > Hardware accelerator > GPU


#### *Step - 3: Checking for GPU and already available python packages*

    !nvidia-smi
    !pip freeze


#### *Step - 4: Uploading Dataset to research folder*

    %cd /content/drive/MyDrive/TFOD1.x/models/research/
    !ls

    !unzip mask_images.zip


#### *Step - 5: Download Model from Model Zoo*

    !wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

    !ls

    !tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz   # Change Folder Name according to the model downloaded
    !mv faster_rcnn_inception_v2_coco_2018_01_28/ faster_rcnn   # Same folder name change goes here


#### *Step - 6: Uninstall TF2.x and install TF1.x*

    # GPU Check
    !nvidia-smi
    # Uninstall Tensorflow 2.4.1
    !pip uninstall tensorflow==2.4.1 -y
    # Install Tensorflow 1.1.4
    !pip install tensorflow-gpu==1.14.0
    #Import Tensorflow
    import tensorflow as tf
    # Tensorflow version check
    print(tf.__version__)
    # GPU Test for TF
    print(tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    ))

    print(tf.test.is_built_with_cuda())


#### *Step - 7: Download generate_tfrecords.py file and upload the same to research folder*

[Download](https://drive.google.com/file/d/1c2Gxf2GpGVPfJn6r9TLpz-H1NdUXj3h6/view?usp=sharing) generate_tfrecords.py

    Upload the file to research folder

    Changes can be made in below files incase multiple classes are present.

    # TO-DO replace this with label map
    # Changes can be made in if else statement according to the dataset
    def class_text_to_int(row_label):
        if row_label == 'with_mask':
            return 1
        elif row_label == 'without_mask':
            return 2
        elif row_label == 'mask_weared_incorrect':
            return 3
        else:
            None


#### *Step - 8: Execute generate_tfrecord.py for test and train data*

    !python generate_tfrecord.py --csv_input=mask_images/train/train.csv --image_dir=mask_images/train/ --output_path=train.record

    !python generate_tfrecord.py --csv_input=mask_images/test/test.csv --image_dir=mask_images/test/ --output_path=test.record

!!! Note
    Check train.record and test.record files are present in research folder.

??? Note "Click here to see full code of generate_tfrecord.py"

        from __future__ import division
        from __future__ import print_function
        from __future__ import absolute_import

        import os
        import io
        import pandas as pd
        import tensorflow as tf

        from PIL import Image
        from object_detection.utils import dataset_util
        from collections import namedtuple, OrderedDict

        flags = tf.app.flags
        flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
        flags.DEFINE_string('image_dir', '', 'Path to the image directory')
        flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
        FLAGS = flags.FLAGS


        # TO-DO replace this with label map
        def class_text_to_int(row_label):
            if row_label == 'with_mask':
                return 1
            elif row_label == 'without_mask':
                return 2
            elif row_label == 'mask_weared_incorrect':
                return 3
            else:
                None


        def split(df, group):
            data = namedtuple('data', ['filename', 'object'])
            gb = df.groupby(group)
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


        def create_tf_example(group, path):
            with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size

            filename = group.filename.encode('utf8')
            image_format = b'jpg'
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []

            for index, row in group.object.iterrows():
                xmins.append(row['xmin'] / width)
                xmaxs.append(row['xmax'] / width)
                ymins.append(row['ymin'] / height)
                ymaxs.append(row['ymax'] / height)
                classes_text.append(row['class'].encode('utf8'))
                classes.append(class_text_to_int(row['class']))

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            return tf_example


        def main(_):
            writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
            path = os.path.join(os.getcwd(), FLAGS.image_dir)
            examples = pd.read_csv(FLAGS.csv_input)
            grouped = split(examples, 'filename')
            for group in grouped:
                tf_example = create_tf_example(group, path)
                writer.write(tf_example.SerializeToString())

            writer.close()
            output_path = os.path.join(os.getcwd(), FLAGS.output_path)
            print('Successfully created the TFRecords: {}'.format(output_path))


        if __name__ == '__main__':
            tf.app.run()


#### *Step - 9: Copy Files faster_rcnn_inception_v2_coco.config and labelmap.pbtxt to training folder in research directory*

[Download](https://drive.google.com/file/d/1mmDdqZFGa_F2FEIsW4GBTae-tqnIM6GY/view?usp=sharing) faster_rcnn_inception_v2_coco.config files  

[Download](https://drive.google.com/file/d/1woz7_I5EiB9UBLnIE3JCqLuldYesdzlm/view?usp=sharing) labelmap.pbtxt

    Upload Both files in training folder inside research directory.

    A total of 7 changes have to be made in .config file. Refer below for details.

??? Note "Click here to see what changes to be done in config file"

        # Faster R-CNN with Inception v2, configuration for MSCOCO Dataset.
        # Users should configure the fine_tune_checkpoint field in the train config as
        # well as the label_map_path and input_path fields in the train_input_reader and
        # eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
        # should be configured.


        model {
        faster_rcnn {
            num_classes: 3  # Change: No of classes to be determined
            image_resizer {
            keep_aspect_ratio_resizer {
                min_dimension: 600
                max_dimension: 1024
            }
            }
            feature_extractor {
            type: 'faster_rcnn_inception_v2'
            first_stage_features_stride: 16
            }
            first_stage_anchor_generator {
            grid_anchor_generator {
                scales: [0.25, 0.5, 1.0, 2.0]
                aspect_ratios: [0.5, 1.0, 2.0]
                height_stride: 16
                width_stride: 16
            }
            }
            first_stage_box_predictor_conv_hyperparams {
            op: CONV
            regularizer {
                l2_regularizer {
                weight: 0.0
                }
            }
            initializer {
                truncated_normal_initializer {
                stddev: 0.01
                }
            }
            }
            first_stage_nms_score_threshold: 0.0
            first_stage_nms_iou_threshold: 0.7
            first_stage_max_proposals: 300
            first_stage_localization_loss_weight: 2.0
            first_stage_objectness_loss_weight: 1.0
            initial_crop_size: 14
            maxpool_kernel_size: 2
            maxpool_stride: 2
            second_stage_box_predictor {
            mask_rcnn_box_predictor {
                use_dropout: false
                dropout_keep_probability: 1.0
                fc_hyperparams {
                op: FC
                regularizer {
                    l2_regularizer {
                    weight: 0.0
                    }
                }
                initializer {
                    variance_scaling_initializer {
                    factor: 1.0
                    uniform: true
                    mode: FAN_AVG
                    }
                }
                }
            }
            }
            second_stage_post_processing {
            batch_non_max_suppression {
                score_threshold: 0.0
                iou_threshold: 0.6
                max_detections_per_class: 100
                max_total_detections: 300
            }
            score_converter: SOFTMAX
            }
            second_stage_localization_loss_weight: 2.0
            second_stage_classification_loss_weight: 1.0
        }
        }

        train_config: {
        batch_size: 1
        optimizer {
            momentum_optimizer: {
            learning_rate: {
                manual_step_learning_rate {
                initial_learning_rate: 0.0002
                schedule {
                    step: 900000
                    learning_rate: .00002
                }
                schedule {
                    step: 1200000
                    learning_rate: .000002
                }
                }
            }
            momentum_optimizer_value: 0.9
            }
            use_moving_average: false
        }
        gradient_clipping_by_norm: 10.0
        fine_tune_checkpoint: "faster_rcnn/model.ckpt"  # Change: Path for model dowloaded
        from_detection_checkpoint: true
        # Note: The below line limits the training process to 200K steps, which we
        # empirically found to be sufficient enough to train the COCO dataset. This
        # effectively bypasses the learning rate schedule (the learning rate will
        # never decay). Remove the below line to train indefinitely.
        num_steps: 20000    # Change: No of iterations for training
        data_augmentation_options {
            random_horizontal_flip {
            }
        }
        }

        train_input_reader: {
        tf_record_input_reader {
            input_path: "train.record"  # Change: Path for train.record
        }
        label_map_path: "training/labelmap.pbtxt"  # Change: Path for labelmap.pbtxt
        }

        eval_config: {
        num_examples: 8000
        # Note: The below line limits the evaluation process to 10 evaluations.
        # Remove the below line to evaluate indefinitely.
        max_evals: 10
        }

        eval_input_reader: {
        tf_record_input_reader {
            input_path: "test.record"  # Change: Path for test.record
        }
        label_map_path: "training/labelmap.pbtxt"  # Change: Path for labelmap.pbtxt
        shuffle: false
        num_readers: 1
        }


#### *Step - 10: Copy _deployment_ and _nets_ folder from _research/slim_ into the _research_ folder*


#### *Step - 11: Start Model Training*

    !python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

!!! Warning

    Few warnings might show, please ignore the same. If model starts training it takes around 40-50 minutes on Google Colab Pro.


#### *Step - 12: Copy export_inference_graph.py from object_detection folder to research folder*

    !python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory mask_model

!!! Note

    Model is now ready to test. Please refer mask_model/frozen_inference_graph.pb which contains trained model.


#### *Step - 13: Run below code and test the images*

    %cd object_detection

    import numpy as np
    import os
    import six.moves.urllib as urllib
    import sys
    import tarfile
    import tensorflow as tf
    import zipfile

    from distutils.version import StrictVersion
    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops

    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
    
    # This is needed to display the images.
    %matplotlib inline

    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # What model to download.
    MODEL_NAME = '/content/drive/MyDrive/TFOD1.x/models/research/mask_model'
    #MODEL_FILE = MODEL_NAME + '.tar.gz'
    #DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/content/drive/MyDrive/TFOD1.x/models/research/training', 'labelmap.pbtxt')


??? Note "TFOD Code"

        detection_graph = tf.Graph()
        with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

        # For the sake of simplicity we will use only 2 images:
        # image1.jpg
        # image2.jpg
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)


        def run_inference_for_single_image(image, graph):
            with graph.as_default():
                with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict


        % matplotlib inline
        for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)

**Sample Output1:**
![Sample Output1](images\sample_img1.PNG)

**Sample Output2:**
![Sample Output2](images\sample_img2.PNG)

**Sample Output3:**
![Sample Output3](images\sample_img3.PNG)