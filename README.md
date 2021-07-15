# football_player_tracking

In this Repo, we have provided the tracking of football players using Background Subtraction and classifying them using a neural network.


## Detection(First Part)

For detecting players we have used [Background Subtraction Methods](https://docs.opencv.org/master/d8/d38/tutorial_bgsegm_bg_subtraction.html) such as MOG2, KNN, etc.

Because every area, of the frame, has different features, such as upper area has a lot of noise and players are smaller than lower areas, ...,  We have split the frame into 3 section:

- Upper Area: which we do [Erosion](https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html) first for removing the Banners and anything the above them.

- Middel Area: we do [Erosion](https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html) first for removing the ball and any other noises. Then we do [Dilation](https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html) for making players bigger. 

- Lower Area: Because in the lower area players are much bigger than two other areas, we do [Closing](https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html).

all you can see in `get_blobs` function in `main.py` file. feel free to change the kernels and see what will happend.


## Classification(Second Part)

For classifying players of the teams and referees, we have used Neural Network. Our framework for building the Neural Network was Tensorflow(Keras). 

Our dataset was from https://datasets.simula.no/alfheim/ that provides us with three points of view from the football field.

We have four different trained models(that you can see in `model_4, model_5, model_6, model_7` folders), each has an accuracy of around 97%, the best of them is `model_4` with the validation accuracy of 98% that use the [lenet 5](https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/#:~:text=The%20Architecture%20of%20the%20Model&text=The%20network%20has%205%20layers,have%20two%20fully%20connected%20layers.) architecture. If you want to change the architecture and see the result, you can change the `build_model` function in `model_architecture.py` file. 



## Start
```bash
    python3 -m venv env # or you can use virtual env but python3 is requred
    source ./env/bin/activate

    pip install -r requirements.txt

    python main.py
```

## Outputs

### sample 1

#### detection
- 
![alt text](./screenshots/detection_1.png)

#### classification

- 
![alt text](./screenshots/classify_1.png)

### sample 2

#### detection

- 
![alt text](./screenshots/detection_2.png)

#### classification
- 
![alt text](./screenshots/classify_2.png)


### sample 3 
-
![alt text](./screenshots/football.gif)
