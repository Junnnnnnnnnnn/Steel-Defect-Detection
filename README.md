# README

## 주제

- 철강 사진을 보고 불량품을 분류하는 웹 페이지

## 나의 역할

### 파트

- 백엔드

### 개발 툴

- Flask Framework
  - 소규모 프로젝트 이고 분석이 python으로 진행되기 때문에 python 언어를 사용한 flask Framework를 사용

## 개발 중 고민했던 점

>**"학습된 모델의 크기가 400MB가 넘었기 때문에 로드하는데 문제가 생김"**

- RESTfull api를 사용하여 /predict를 호출할때 마다 load_model을 사용할려고 했으나 파일 용량이 크기 때문에 사용에 불편을 줌
  - 해결 : 서버 실행시 load_model을 통해 세션에 모델을 저장 함으로서 시간적 에로점을 해결할 수 있음

```python
from tensorflow.python.keras.backend import set_session

a = tf.ConfigProto()
a.gpu_options.allow_growth=True

sess = tf.Session(config=a)
graph = tf.get_default_graph()

def load_model():
    global model
    global sess
    set_session(sess)
    severstal_config = SeverstalConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir='static/logs', config=severstal_config)
    model.load_weights("static/model/Mask_RCNN/mrcnn_0319.h5")
    print("load_model() 실행")
```

- tensorflow에서 제공하는 Session메서드를 가져와서 load_model에서 session을 지정해주면 모델이 session에 올라가는 것 같다.. 자세히는 모르겠다 stackoverflow에 있는 내용인지라.. 따로 session 공부를 해야겠다.

```python
if __name__ == '__main__':
    #global severstal_config
    load_model()
    app.run(debug=True,host="0.0.0.0")
```

- 다음 과 같이 서버가 시작 될때 정의 해놓은 load_model()을 호출한다.

```python
# 철강사진 분석결과
@app.route('/predict' , methods=["POST"])
def predict():
    #그래프가 뭘까..
    # with tf.device(DEVICE):
    origin_img=skimage.io.imread('static/img/uploaded_img.jpg')
    class_names=['1','2','3','4']*30
    img = np.reshape(origin_img,(1, origin_img.shape[0], origin_img.shape[1], 3) )
    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        results = model.detect(img, verbose=1)
        r = results[0]
        print(r)

    
    image_name = 'result_img'
    visualize.save_image(origin_img, image_name, r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, mode=0)
    #class_ids 값을 중복된 result값을 하나로
    r_class = np.unique(r['class_ids'])
    print("=============================")
    print("클레스 넘버는" , r['class_ids'])
    print("=============================")
    return render_template("result.html",class_num=r_class)
```

- 고드 중간에 보면 sess와 gragh를 글로벌 변수로 받아 오고 다음과 같이 써주면 with내에 model.detect을 사용하면 불러 놓은 model을 사용 할 수 있다.

>**"static 폴더를 서버에 올릴때 그 폴더 내용이 캐시화 되어 바뀐 이미지를 로드 하였지만 이전 이미지가 출력됨"**

- cache의 init 설정을 no-cache로 설정해 줘야 한다

```python
#cache disable init code
@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r
```

> "mrcnn을 사용해서 detecting 된 철강 결함 사진을 save_img할때 어떻게 해야하는가?"

- mrcnn에서 제공해주는 visualize.py 코드에 save_img메서드를 만들어 주면 된다.

```python
def save_image(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
               scores_thresh=0.1, save_dir=None, mode=0):
    """
        image: image array
        image_name: image name
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        save_dir: (optional) the path to store image
        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
    mode_list = [0, 1, 2, 3]
    assert mode in mode_list, "mode's value should in mode_list %s" % str(mode_list)
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    useful_mask_indices = []
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        # return 
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue
        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        useful_mask_indices.append(i)
    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        # return 
    colors = random_colors(len(useful_mask_indices))
    if mode != 3:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)
    if mode != 1:
        for index, value in enumerate(useful_mask_indices):
            masked_image = apply_mask(masked_image, masks[:, :, value], colors[index])
    masked_image = Image.fromarray(masked_image)
    if mode == 3:
        masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
        return
    draw = ImageDraw.Draw(masked_image)
    colors = np.array(colors).astype(int) * 255
    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        score = scores[value]
        label = class_names[class_id]
        y1, x1, y2, x2 = boxes[value]
        if mode != 2:
            color = tuple(colors[index])
            draw.rectangle((x1, y1, x2, y2), outline=color)
        # Label
        font = ImageFont.truetype('arial.ttf', 15)
        draw.text((x1, y1), "%s %f" % (label, score), (255, 255, 255), font)
    masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
```

- 위 코드는 stackoverflow에 기제된 함수로 매개변수로 이미지에 대한 정보를 입력하면 사용 가능하다

## 참고 사이트

https://stackoverflow.com/

https://google.com/

