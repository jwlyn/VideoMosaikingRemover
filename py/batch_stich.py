
import glob
import os

import cv2
from stich import Stitcher, Method


def main():
    import time
    # main()
    os.chdir(os.path.dirname(__file__))

    number = 19
    file1 = "../resource/{}-right*.jpg".format(number)
    file2 = "../resource/{}-left.jpg".format(number)

    start_time = time.time()
    try:
        for method in (Method.ORB, Method.SIFT):

            for f in glob.glob(file1):
                print(f, method)
                name = f.replace('right', method.name)
                # print(file2, name)

                img2 = cv2.imread(file2)
                img1 = cv2.imread(f)
                stitcher = Stitcher(img1, img2, method=method)
                stitcher.stich(show_result=False)
                cv2.imwrite(name, stitcher.image)
                print("Time: ", time.time() - start_time)
                # print("M: ", stitcher.M)
    except Exception as e:
        print("Error!: ", e)
    print('\a')


if __name__ == "__main__":
    main()
