# PoC: Attention Tracking using Facial Recognition for an online class

## Description

We use facial recognition and pose estimation to determine whether the user is paying attention to their screen and when they are looking away.


## Build

```
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

## Use

 - Start Attention Tracking
     - `source env/bin/activate; python track_attention.py`

 - Show/Hide Information
     - Press `s` key to toggle.

 - Exit
     - Press `Crtl + C`


## Refrences

  - Tools
      - `Python`
      - `OpenCV`
      - `TensorFlow`
      - `Numpy`

  - Facial Landmark Model
      - `https://github.com/yinguobing/cnn-facial-landmark`
  
