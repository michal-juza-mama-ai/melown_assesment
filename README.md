# melown_assesment
I made my life a bit easier:
  1) I assume that all the images are taken from the same height (no resizing)
  2) As I've rather older notebook without CUDA available
    a) I limited my test/train set to some feasible size - there could be additional transforms to generate more training data
    b) I took existing segmentation pytorch model (FCN on the resnet101 backbone) and only trained the new classifier
    
  


![image](https://user-images.githubusercontent.com/75313287/114025149-1e302780-9875-11eb-938c-1e6f6e0d7f02.png)
