# melown_assesment
I made my life a bit easier:
  1) I assume that all the images are taken from the same height (no resize involved)
  2) As I've rather older notebook without CUDA available
    a) I limited my test/train set to some feasible size - there could be additional transforms to generate more training data
    b) I took existing segmentation pytorch model (FCN on the resnet101 backbone) and only trained the new classifier
