from isaac_app import create_isaac_app, start_isaac_app, create_sample_bridge
import time
import cv2
import numpy as np
isaac_app = create_isaac_app(filename="/mnt/isaac_2019_2/apps/carter_sim_struct2depth/apps/carter_sim.app.json")

start_isaac_app(isaac_app)

bridge = create_sample_bridge(isaac_app)

while True:

    while True:
        if bridge.get_sample_count() >= 1:
            break
        print('Waiting for samples')
        time.sleep(1)

    samples = bridge.acquire_samples(1)
    print(samples[0][1])
    # cv2.imshow('b',samples[0][0])
    # cv2.waitKey(500)
    # for image, mask in samples:
    #     cv2.imshow('image', image / 255.0)
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(500)
    # # cv2.imshow('mask', mask[0][0])
    # # cv2.waitKey(500)
    # print(samples)