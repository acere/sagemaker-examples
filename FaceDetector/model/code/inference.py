import json
import logging

import cv2
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import numpy as np
from skimage import transform as trans

try:
    from .mtcnn_detector import MtcnnDetector
except ImportError:
    from mtcnn_detector import MtcnnDetector

logging.basicConfig(level=logging.DEBUG)


def model_fn(model_dir):
    """
    Load the onnx model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model
    """
    if len(mx.test_utils.list_gpus()) == 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)

    det_threshold = [0.6, 0.7, 0.8]
    detector = MtcnnDetector(
        model_folder=f"{model_dir}/mtcnn-model",
        ctx=mx.cpu(),
        num_worker=1,
        accurate_landmark=True,
        threshold=det_threshold,
    )
    image_size = (112, 112)
    sym, arg_params, aux_params = onnx_mxnet.import_model(f"{model_dir}/resnet100.onnx")
    # create module
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(
        for_training=False, data_shapes=[("data", (1, 3, image_size[0], image_size[1]))]
    )
    mod.set_params(arg_params=arg_params, aux_params=aux_params)

    return (detector, mod)


def transform_fn(mod, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param mod: The super resolution model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    (detector, arcface) = mod
    input_data = np.array(json.loads(data), dtype="uint8")
    ret = detector.detect_face(input_data, det_type=0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0] == 0:
        return None
    bbox = bbox[0, 0:4]
    points = points[0, :].reshape((2, 5)).T

    # Call preprocess() to generate aligned images
    nimg = preprocess(input_data, bbox, points, image_size="112,112")
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    arcface.forward(db, is_train=False)
    embedding = arcface.get_outputs()[0].asnumpy()
    embedding /= (embedding ** 2).sum() ** 0.5
    return (
        json.dumps(
            {
                "preprocessed_image": aligned.tolist(),
                "feature_vector": embedding.flatten().tolist(),
            }
        ),
        output_content_type,
    )


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get("image_size", "")
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(",")]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get("margin", 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1] : bb[3], bb[0] : bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


if __name__ == "__main__":
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    base_path = Path(__file__).parent
    content_type = "application/json"
    net = model_fn(base_path.parent)

    def invoke_prediction(data_path):
        data = cv2.imread(data_path)
        data = json.dumps(data.tolist())
        out, _ = transform_fn(net, data, content_type, content_type)
        out = json.loads(out)
        return out["preprocessed_image"], np.array(out["feature_vector"], dtype=float)

    data_path1 = (base_path / "../../images/player1.jpg").resolve().as_posix()
    data_path2 = (base_path / "../../images/player2.jpg").resolve().as_posix()

    face1, out1 = invoke_prediction(data_path1)
    face2, out2 = invoke_prediction(data_path2)

    # Compute squared distance between embeddings
    dist = np.sum(np.square(out1 - out2))
    # Compute cosine similarity between embedddings
    sim = np.dot(out1, out2.T)

    # Print predictions
    print(f"Distance = {dist:.4f}")
    print(f"Similarity = {sim:.4f}")

    plt.imsave(
        (base_path / "../../player1_preprocessed.jpg").resolve(),
        np.transpose(np.array(face1, dtype="uint8"), (1, 2, 0)),
    )
    plt.imsave(
        (base_path / "../../player2_preprocessed.jpg").resolve(),
        np.transpose(np.array(face2, dtype="uint8"), (1, 2, 0)),
    )

    print("Run successful")
