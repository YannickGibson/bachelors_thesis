# Real-time Object Detection for ping-pong
This project focuses on identifying specific targets in a ping-pong match. Among these targets are ping-pong paddles and players. Furthermore, we also decided to detect a ball and a scorekeeper in matches. 

We applied a computer vision system in the ubiquitous Python programming language for object detection with the architecture **YOLOv8** (You Only Look Once version 8) based on YOLOv5 paper. 

This project gets a video input, draws the enclosing bounding boxes around objects of interest, and displays the video with predictions. We acquire unlabeled data and annotate it manually while also utilizing the pre-annotation method with a pre-trained model. In addition, we supply a plethora of data manipulation techniques and analysis of our results. We end with a robust model detecting all four defined classes at the inference speed of 72 Frames Per Second (FPS).

Detected objects:
-   paddles
-   players
-   scorekeeper
-   ball

This is a git repository, hence the model files, annotations and datasets are not supplied here.


### Video example on unseen data (120 FPS 50% speed using YOLOv8m (medium)):


https://github.com/YannickGibson/bachelors_thesis/assets/57909721/0ff3d92d-7134-4438-a0c2-cd3dcbf70e33


# Run locally

### Establish an environment
```
conda create -n ping_pong python==3.11
conda activate ping_pong
```
### Install dependencies
```
pip install -r requirements.txt
pip install . -e
```
### Run model
```
python play_model.py
```

### Start Label Studio
```
make sl
```

