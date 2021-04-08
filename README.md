# melown_assesment
I made my life a bit easier:
  1) I assume that all the images are taken from the same height (no resizing)
  2) As I've rather older notebook without CUDA available
    a) I limited my test/train set to some feasible size - there could be additional transforms to generate more training data
    b) I took existing segmentation pytorch model (FCN on the resnet101 backbone) and only trained the new classifier
    
  3) For the training I randomly cropped images (224x224) from the provided set (see  ``generate_samples2()`` in ``mj_utils.py``
  4) I re-used some existing code for the ``SegmentationDataset`` and modified it see ``segdataset.py`` for details) - this class is later used for the training
  5) Training itself can be found in ``mj_trainer.py`` and it is used by ``mj_project.py`` which is actually script defining the training. I've not trained model from scratch but I used the existing ``FCNResnet101`` from Pytorch and only retrain the classifier for the required classes
  6) Function to perform segmentation can be found in ``mj_predict.py`` (``segment_image()``) - example of result you can see below

I admit that the example bellow is not fair as the training samples were taken from this image as well :). As the training is very slow on my computer the resulting model is definitelly not perfect and is only to demonstrate that it somehow works:). 
  


![image](https://user-images.githubusercontent.com/75313287/114025149-1e302780-9875-11eb-938c-1e6f6e0d7f02.png)
