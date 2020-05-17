Run stitch.py to generate the output
cd video-object-removal
cd get_mask
bash make.sh
cd ../inpainting
bash install.sh
cd ..

```python
matcher = Matcher(img1, img2, Method.SIFT)
matcher.match(show_match=True)
sticher = Sticher(img1, img2, matcher)
sticher.stich()
```

It is divided into two parts, `Matcher` and` Sticher`, which are used for image content recognition and image splicing respectively

## Introduction to Matcher

### Constructor

```python
class Matcher():

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SURF, threshold=800) -> None:

        ...
```

This type is used to input two images and calculate their eigenvalues. The two images are images in the numpy array format. The parameter of method requires SURF, SIFT or ORB, and the parameter of threshold is the detection of eigenvalues. The required threshold.

### Characteristic value calculation

```python
    def compute_keypoint(self) -> None:
        """计算特征点

        Args:
            image (np.ndarray): 图像
        """
        ...
```
Use the given eigenvalue detection method to detect the eigenvalue of the image.

### match

```python
    def match(self, max_match_lenth=20, threshold=0.04, show_match=False):
        ...
```

To match the calculated feature values of two pictures, for ORB, OpenCV's `BFMatcher` algorithm is used, and for other feature detection methods,` FlannBasedMatcher` algorithm is used.

## Sticher introduction

### Constructor

```python
class Sticher:

    def __init__(self, image1: np.ndarray, image2: np.ndarray, matcher: Matcher):

        ...
```

Input the image and match to splice the image. At present, simple matrix matching and average stitching are used.

### Flatten

```python
    def stich(self, show_result=True, show_match_point=True):
        """对图片进行拼合
            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        ...
```

Put two images together, use perspective transformation matrix, and use the average value to seamlessly join the pictures.

### Fusion

```python
    def blend(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:

        ...
```

The simple average method is currently used.

### Helper function

#### Average

```python
    def average(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarry:

        ...
```

Returns the average of two pictures.

#### Boundary calculation

```python
    def get_transformed_size(self) ->Tuple[int, int, int, int]:

        ...
```

Calculate the deformed boundary, so as to shift the picture accordingly to ensure that all the images appear on the screen.

#### Coordinate transformation

```python
    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float=None, M=None) -> Tuple[float, float]:

        ...
```

Find the new coordinates of a point under the transformation matrix (self.M). If the parameter `M` is selected, use M to perform the coordinate transformation operation.