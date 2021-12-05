# hand-mouse

**Personal project at Baram, Academic group**</br>
**Division of Robotics, Kwangwoon University at Seoul, Korea**</br>
**Term:** 03.2020. ~ 07.2020.</br>
**Intro:** Virtual Mouse using Hand detection

**OS:** Ubuntu 18.04</br>
**Hardware spec:** GTX1070, CUDA 10.0, cuDNN 7.5.0</br>
**Programming Language:** Python

\* **Origin source:** This project is based on [here](https://github.com/MrEliptik/HandPose)

---

Used Tensorflow API for training and detecting my hand. Made the hand dataset by myself, and each hand command is from sign language for alphbet which have certain features.

By cam image, detect my hand first, and classify hand shape. Mouse pointer is tracking my hand and each hand shape have individual input command, like left click, right click, esc, etc. With those command, perform virtual mouse or keyboard input. For these automatics, I used python library, pyautogui.

**My hand shape commands:** 

![txt](/my_recognize_handpose/images/txt.jpg)
![commands](/my_recognize_handpose/images/my_handpose.png)

---

![video](/my_HandPose/Results/4.gif)

---

My epilogue about this project is [here](https://msjun23.github.io/projects/Project-2020-1-Hand-Mouse/)