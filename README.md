# StyleGAN (ver1)

- GPU: RTX 3090Ti
- Dataset: [ffhq 512x512](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)
- Time: 10 days


## prediction_images
-----------------------
- 256x256 해상도의 최종 결과 이미지이다. (목표는 512x512 해상도였지만, 예상 학습 시간을 계산해보았을 때 30일이 걸려서 256x256에서 그만두었다.)
![prediction_image(256x256)](https://user-images.githubusercontent.com/66504341/217241255-3de5bc52-80b8-401d-9486-7c3a6c35c8ac.jpg)


## blob-like artifact
------------------------
- 이 문제는 논문의 설명대로 64x64부터 나타나기 시작했다. 아래는 64x64 해상도의 이미지이다.
![prediction_image(64x64)](https://user-images.githubusercontent.com/66504341/217241262-849a99a1-0bdf-4a41-8dfd-8a49c342e9d2.jpg)    
------------------------
    
- 아래는 128x128 해상도의 이미지이다.
![prediction_image(128x128)](https://user-images.githubusercontent.com/66504341/217241258-632e8fba-8964-450b-8a3d-edc640531cd1.jpg)


--------------------
#### StyleGAN2를 이용하여 개선을 해보는 것이 다음 목표이다.