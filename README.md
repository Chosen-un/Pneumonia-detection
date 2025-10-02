# Pneumonia Detection from Chest X-Rays using Deep Learning

This project uses deep learning techniques, specifically transfer learning with convolutional neural networks (CNNs), to accurately classify chest X-ray images as either Normal or indicating Pneumonia.

The primary goal is to compare several state-of-the-art CNN architectures, select the best-performing one, and fine-tune it to achieve high classification accuracy. The entire workflow is optimized for performance on a GPU, such as the T4 available in Google Colab.


## Project Overview
Pneumonia remains a significant global health issue, and rapid, accurate diagnosis is crucial for effective treatment. This project leverages a publicly available dataset of chest X-ray images to build a robust diagnostic tool.

**The core approach involves:**

**Data Preprocessing:** An efficient tf.data pipeline is used to load, augment, and preprocess images.

**Transfer Learning:** Five powerful, pre-trained models are used as a feature extraction base:

- VGG16

- MobileNetV2

- DenseNet121

- InceptionV3

- ResNet50

**Model Comparison:** The models are trained and compared to identify the most effective architecture for this specific task.

**Fine-Tuning:** The best-performing model is further trained (fine-tuned) to improve its accuracy.

**Optimization:** Techniques like mixed-precision training and on-GPU data augmentation are used to ensure the process runs efficiently on a GPU.


## Methodology 

### Data Pipeline

An optimized data pipeline was created using tf.data.image_dataset_from_directory for efficient loading. To prevent the CPU from becoming a bottleneck for the GPU, the following optimizations were implemented:

- cache(): Caches the dataset in memory after the first epoch.

- prefetch(): Prepares subsequent batches of data in the background while the GPU is training on the current batch.

- On-GPU Augmentation: A keras.Sequential model with layers like RandomFlip and RandomRotation was used to perform data augmentation directly on the GPU, which is significantly faster than traditional CPU-based methods.

### Initial Model Training (Transfer Learning)

Each of the five pre-trained models was loaded with its ImageNet weights. The original convolutional base was frozen (made non-trainable), and a new classification head was added on top. This new head consists of:

- GlobalAveragePooling2D layer

- Two Dense layers with Dropout and L2 regularization to prevent overfitting.

- A final softmax output layer.

To handle the class imbalance in the dataset, class weights were calculated and applied during training, ensuring the model did not become biased towards the majority class.

### Fine-Tuning

After the initial comparison, the best model (MobileNetV2) was selected for fine-tuning. This process involved:

- Unfreezing the top 30% of the layers in the pre-trained base model.

- Re-compiling the model with a very low learning rate (1×10 
−5
 ) to make small, careful adjustments to the unfrozen weights.

- Continuing training on the full dataset with more aggressive data augmentation.

This two-stage process allows the model to first learn high-level features from the new data and then gently adjust its more specific, low-level feature detectors for the X-ray domain.

## Results
Model Comparison

The initial training run on 50% of the data yielded the following validation accuracies, identifying MobileNetV2 as the top performer.

| Model	  |  Validation Accuracy | Accuracy |
|----------|---------------------|----------|
|MobileNetV2	|87.02%|92.74%|
|DenseNet121	|85.90%|87.44%|
|VGG16	|85.42%|89.47%|
|InceptionV3	|79.33%|89.96|
|ResNet50	|77.24%|60.54|

Fine-Tuning and Final Performance

The MobileNetV2 model was then fine-tuned. During this phase, the model began to overfit—where training accuracy increased while validation accuracy decreased. However, the EarlyStopping callback successfully halted the training and restored the weights from the best-performing epoch.

The final results for the fine-tuned MobileNetV2 model are:

- Final Validation Accuracy: 81.09%
- Final Accuracy: 93.37%

While the final validation accuracy is slightly lower than the peak from the initial training, this is a more robust result obtained from training on the full dataset with a more complex (partially unfrozen) model. The process demonstrates a complete and realistic deep learning workflow.



