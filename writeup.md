# Project: Perception Pick & Place
[//]: # (Image References)

[confusion-matrix-unnormalized]: ./misc_images/confusion-matrix-unnormed.png
[confusion-matrix-normalized]: ./misc_images/confusion-matrix-normed.png
[train-svm-result]: ./misc_images/train-svm-result.png
[pr2-point-cloud]: ./misc_images/pr2-point-cloud.png
[pass-through-filtering]: ./misc_images/pass-through-filtering.png
[noise-filtering]: ./misc_images/noise-filtering.png
[ransac-table]: ./misc_images/ransac-table.png
[ransac-objects]: ./misc_images/ransac-objects.png
[clustering]: ./misc_images/clustering.png
[world-1-1]: ./misc_images/object-recognition-world-1-1.png
[world-1-2]: ./misc_images/object-recognition-world-1-2.png
[world-1-3]: ./misc_images/object-recognition-world-1-3.png
[world-2-1]: ./misc_images/object-recognition-world-2-1.png
[world-2-2]: ./misc_images/object-recognition-world-2-2.png
[world-2-3]: ./misc_images/object-recognition-world-2-3.png
[world-2-4]: ./misc_images/object-recognition-world-2-4.png
[world-3-1]: ./misc_images/object-recognition-world-3-1.png
[world-3-2]: ./misc_images/object-recognition-world-3-2.png
[world-3-3]: ./misc_images/object-recognition-world-3-3.png
[world-3-4]: ./misc_images/object-recognition-world-3-4.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The project starts by having a point cloud that are filled with noise:

![PR2 View without Filtering][pr2-point-cloud]

The first thing that `pcl_callback()` does is that it executes pass through filter, so that only the table plane and the objects above it being included in the point cloud that is passed through the perception pipeline. Here is the code responsible in doing the pass through filtering:

```python
# TODO: PassThrough Filter
passthrough = cloud_filtered.make_passthrough_filter()

filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()

# Second pasthrough for the table edge
passthrough = cloud_filtered.make_passthrough_filter()

filter_axis = 'x'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.33
axis_max = 1.0
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()
```

There are two pass through filterings. The first one is to output only the table and the objects on it, so the code filters through the Z axis. The second one is to filter the table's edge that is parallel to the YZ plane, so the code filters through the X axis. Here is the result of the code:

![Pass Through Filtering Result][pass-through-filtering]

Following the pass through filtering is the noise filtering using the following code:

```python
# TODO: Statistical Outlier Filtering
outlier_filter = cloud_filtered.make_statistical_outlier_filter()

# Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(50)

# Set threshold scale factor
x = 0.4

# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)

cloud_filtered = outlier_filter.filter()
```

This will filter the noise in the point cloud and resulting in the point cloud as follow:

![Noise Fitering Result][noise-filtering]

Next, what `pcl_callback()` does is executing RANSAC plane segmentation to grab only the objects point cloud. Here is the code responsible in doing such thing:

```python
# TODO: RANSAC Plane Segmentation
seg = cloud_filtered.make_segmenter()

seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

max_distance = 0.01
seg.set_distance_threshold(max_distance)

inliers, coefficients = seg.segment()

# TODO: Extract inliers and outliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
```

The resulting point clouds of the objects and the table are separately captures in the images below:

![RANSAC Objects][ransac-objects]
![RANSAC Table][ransac-table]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

The next step is executing the clustering process to the segmented point cloud to determine different object in the point cloud. The clustering method applied in the `pcl_callback()` is the Euclidean Clustering and the following is the code to apply it:

```python
# TODO: Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
tree = white_cloud.make_kdtree()

# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.05)
ec.set_MinClusterSize(5)
ec.set_MaxClusterSize(5000)

# Search the k-d tree for clusters
ec.set_SearchMethod(tree)

# Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()
```

After obtaining clustered point clouds, `pcl_callback()` assigns a color to each clustered object in the scene and returning a point cloud that includes all clustered objects with certain color applied to them. Here are the code responsible in doing it:

```python
# Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))

color_cluster_point_list = []

for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                         white_cloud[indice][1],
                                         white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])

# Create new cloud containing all clusters, each with unique color
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
```

Finally, the following is the image capturing the result of clustering that shows different object under different color:

![Euclidean Clustering Result][clustering]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

Next up in the perception pipeline is the process of object recognition. This is done by first obtaining training features and training the model in the Exercise 3 project. Several notable functions are `compute_color_histograms()` and `compute_normal_histograms()`. Both of them are using the 16 bins of normalized histogram feature. What differs is that one of them returns HSV histogram while the other one returns surface normals histogram. Here are how the functions are defined under `features.py`:

```python
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # TODO: Compute histograms
    nbins = 16
    bins_range = (0, 256)

    h_hist = np.histogram(channel_1_vals, nbins, bins_range)
    s_hist = np.histogram(channel_2_vals, nbins, bins_range)
    v_hist = np.histogram(channel_3_vals, nbins, bins_range)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    # normed_features = np.random.random(96) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    nbins = 16
    bins_range = (0, 256)

    x_hist = np.histogram(norm_x_vals, nbins, bins_range)
    y_hist = np.histogram(norm_y_vals, nbins, bins_range)
    z_hist = np.histogram(norm_z_vals, nbins, bins_range)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    # normed_features = np.random.random(96)

    return normed_features
```

Then, to obtain training features, the following code is written under `capture_features.py`:

```python
if __name__ == '__main__':
    rospy.init_node('capture_node')
    models = [\
       'sticky_notes',
       'book',
       'snacks',
       'biscuits',
       'eraser',
       'soap2',
       'soap',
       'glue']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(100):
            # make one hundred attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))
```

The above code will output `training_set.sav` on which `train_svm.py` will make a model out of. Here are the results displayed after training the model:

![Train SVM Result][train-svm-result]
![Confusion Matrix not Normalized][confusion-matrix-unnormalized]
![Confusion Matrix Normalized][confusion-matrix-normalized]

Afterwards `pcl_callback()` uses the model to predict the objects exist on the table by using the following code:

```python
# Classify the clusters! (loop through each detected cluster one at a time)
detected_objects_labels = []
detected_objects = []

for index, pts_list in enumerate(cluster_indices):
    # Grab the points for the cluster
    pcl_cluster = extracted_outliers.extract(pts_list)

    # TODO: convert the cluster from pcl to ROS using helper function
    ros_cluster = pcl_to_ros(pcl_cluster)

    # Compute the associated feature vector
    chists = compute_color_histograms(ros_cluster, using_hsv=True)
    normals = get_normals(ros_cluster)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))

    # Make the prediction
    prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
    label = encoder.inverse_transform(prediction)[0]
    detected_objects_labels.append(label)

    # Publish a label into RViz
    label_pos = list(white_cloud[pts_list[0]])
    label_pos[2] += .4
    object_markers_pub.publish(make_label(label,label_pos, index))
```

Here are the results of object recognition under world 1, 2, & 3 respectively:

#### World 1

![World 1 Object Recognition #1][world-1-1]
![World 1 Object Recognitino #2][world-1-3]

#### World 2

![World 2 Object Recognition #1][world-2-1]
![World 2 Object Recognition #2][world-2-2]
![World 2 Object Recognition #3][world-2-3]

#### World 3

![World 3 Object Recognition #1][world-3-1]
![World 3 Object Recognition #2][world-3-2]
![World 3 Object Recognition #3][world-3-3]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

Besides `pcl_callback()`, one more task included in the `project_template.py` is to return output YAML files containing valid `PickPlace` requests. The following are the code to execute that task, which is inside the `pr2_mover()` function:

```python
def pr2_mover(object_list):
    # TODO: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 3

    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    dict_list = []
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    yaml_filename = 'output_{}.yaml'.format(test_scene_num.data)

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)
        centroids.append((np.asscalar(centroid[0]),
                          np.asscalar(centroid[1]),
                          np.asscalar(centroid[2])))

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for idx, object in enumerate(object_list_param):
        object_name.data = object['name']
        object_group = object['group']
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        for indice, label in enumerate(labels):
            if label == object_name.data:
                pick_pose.position.x = centroids[indice][0]
                pick_pose.position.y = centroids[indice][1]
                pick_pose.position.z = centroids[indice][2]

        for indice, box in enumerate(dropbox_param):
            # TODO: Create 'place_pose' for the object
            if box['group'] == object_group:
                place_pose.position.x = box['position'][0]
                place_pose.position.y = box['position'][1]
                place_pose.position.z = box['position'][2]

                # TODO: Assign the arm to be used for pick_place
                arm_name.data = box['name']

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(yaml_filename, dict_list)
```

Moreover, here are several images showing label markers above clustered objects in RViz under different world scene:

#### World 1

![World 1 Labelled Clustered Objects][world-1-2]

#### World 2

![World 2 Labelled Clustered Objects][world-2-4]

#### World 3

![World 3 Labelled Clustered Objects][world-3-4]

---

In conclusion, here are the perception pipeline methods that are implemented in `project_template.py`:
- Point cloud with 5 milimeter voxel
- Two passthrough filtering: z & x axis
- Noise filtering by filtering outliers
- RANSAC place segmentation with 1 centimeter maximum distance parameter
- Euclidean clustering
- Object recognition:
  - RBF Kernel SVM model
  - 100 poses per object for training features
  - Normalized HSV histogram with 16 bins for one feature
  - Normalized surface normals histogram with 16 bins for another feature

For future improvement, the code should publish a point cloud for 3D collision map, so that the robot would avoid collision to certain objects in the scene. Also, the code should be able to publish a value to make the robot rotates, so that it would recognize all the collision objects, like the side tables, in the scene.

Further, the code should also be able to adjust the place pose after placing each object, so that the next object placed in the dropbox would not stack on top of each other and risk falling out of the dropbox.

