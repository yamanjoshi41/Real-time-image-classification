#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow opencv-python numpy matplotlib')


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


# Load CIFAR-10 (or use tf.keras.utils.image_dataset_from_directory for custom data)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[4]:


# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[5]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 classes in CIFAR-10
])


# In[6]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# In[ ]:


# Predict on a test image
img = test_images[0]
pred = model.predict(np.expand_dims(img, axis=0))
predicted_class = class_names[np.argmax(pred)]


# In[ ]:


plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.show()


# In[ ]:


get_ipython().system('pip install ultralytics')


# In[ ]:


from ultralytics import YOLO
import cv2


# In[ ]:


# Load YOLOv8 model (automatically downloads pretrained weights)
model = YOLO("yolov8n.pt")  # 'nano' version (smallest)


# In[ ]:


# Detect objects in an image
results = model("/Users/yamanjoshi/Downloads/pexels-pixabay-104827.jpg")  # Replace with your image path


# In[ ]:


# Show results
results[0].show()


# In[ ]:


#Real time web detection
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw boxes
    
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

