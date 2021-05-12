# Tensorflow Object Detection v1.x
----
The project demonstrates setup and installation of Tensorflow Object Detection version 1.x.  
Please find Colab Notebook [github](https://github.com/Rahulgarg95/tfod1/blob/main/TFOD1_Setup.ipynb).

## Commands
----

* `conda create -n your_env_name python=3.6 -y`:    Create a new Conda Env
* `conda env list`:     Listing all the conda env on Local PC.
* `conda activate your_env_name`:   Activate existing conda env

## Installation Steps (Google Colab):
----

#### *Step - 1: Create a new notebook with Google Colab.*

    1. Visit Colab Website -> https://colab.research.google.com/notebooks/intro.ipynb
    2. File > New notebook

#### *Step - 2: Change runtime to GPU*

    Runtime > Change runtime type > Hardware accelerator > GPU


#### *Step - 3: Checking for GPU and already available python packages*

    !nvidia-smi
    !pip freeze

#### *Step - 4: Uninstall Tensorflow Version 2.x*

    !pip uninstall tensorflow==2.4.1    # Uninstalling TF2.x


#### *Step - 5: Mount Google Drive on Colab*

    from google.colab import drive
    drive.mount('/content/drive')   # Mounting drive on google colab



#### *Step - 6: Installing tensorflow 1.x version with GPU*

    !pip install tensorflow-gpu==1.14.0
    import tensorflow as tf     # Some warnings will generate ignore the same
    print(tf.__version__)       # Checking Tensorflow Version


#### *Step - 7: Checking GPU status and CUDA availability*

    tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
    )

    tf.test.is_built_with_cuda()


#### *Step - 8: Create a new folder and navigate to the same*

    %cd /content/drive/MyDrive/TFOD1.x
    !pwd


#### *Step - 9: Downloading Tensorflow Models Repo*

    !wget https://github.com/tensorflow/models/archive/v1.13.0.zip
    !ls -lrth   # Check if the files are downloaded
    !unzip v1.13.0.zip  #Unzipping the downloaded folder
    !mv models-1.13.0/ models   # Rename models-1.13.0 to models
    %cd models/research


#### *Step - 10: Converting protos file to python interpretor readable format*

    %cd /content/drive/My Drive/TFOD1.x/models/research
    !protoc object_detection/protos/*.proto --python_out=.


#### *Step - 11: Installing the required packages*

    %cd /content/drive/MyDrive/mytest/models/research
    !python setup.py install    # Install object detection


#### *Step - 12: Imporing Required Packages*

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


#### *Step - 13: Importing utils module*

    from utils import label_map_util
    from utils import visualization_utils as vis_util


#### *Step - 14: Downloading different models from Model Zoo*

    #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md  #Link For Model Zoo Repository
    #http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    #MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


#### *Step - 15: Downloading required model and load model graph .pb file*

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


!!! Note
    All images for which object detection needs to be triggered are present on path: `object_detection/test_images`*

#### *Step - 16: Finally loading images and perform your first object detection*

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    # Incase More images are added changes to be made here accordingly.
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

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
    plt.figure(figsize=(30,40))
    plt.imshow(image_np)


Final Output:
![Final Output](images\object_detection_default.PNG)

## Installation Steps (Local PC):
----

#### *Step - 1: Setup conda env (Refer commands section)*

#### *Step - 2: Installed required packages*

    pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
    conda install -c anaconda protobuf

#### *Step - 3: Then follow the same steps for running on google colab from step - 8 (except step-11 as all required modules are installed)*

#### *Step - 4: Code should be now running*

#### *Step - 5: Running the code in realtime with opencv*

    import cv2
    cap = cv2.VideoCapture(0)
    with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
        ret, image_np = cap.read()
        #Expands dimensions since the model expects images to have a shape [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        #Each box represents a part of image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #Actual detection
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        #Visualization of results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates = True,
            line_thickness = 8)
        
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


## Important Files:

* `frozen_interference_graph.pb`
* `object_detection_tutorial.ipynb`
* `mscoco_label_map.pbtxt`: models\research\object_detection\data
* `*.config files`: \models\research\object_detection\samples\configs