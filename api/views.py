import cv2 as cv
from django.shortcuts import render
from django.core.files.storage import default_storage
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pathlib
import pickle
from PIL import Image
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, permissions
import os
import base64
import uuid
from django.core.files.base import ContentFile

if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Method called to convert the base64 encoded


def to_internal_value(data):
    # print('data', data)

    # extracting the base4 string value
    _format, str_img = data.split(';base64,')
    # print('format', _format)
    # print('str_img', str_img)
    # print('type str_img', type(str_img))

    # converting the extracted string data
    decoded_file = base64.b64decode(str_img)
    # print('decoded file', decoded_file)
    # print('type decoded_file', type(decoded_file))

    # taking the first 10 elements of the returned value
    fname = f"{str(uuid.uuid4())[:10]}.png"
    # print('fname', fname)

    return ContentFile(decoded_file, name=fname)


@api_view(['POST'])
@permission_classes((permissions.AllowAny,))
def result(request):
    """
    Return inference
    """
    # print(request.data["image"])

    '''
    Checking for request with encoded data by decoding and comparing the decoded value to its
    original "encoded" base64 value. This check is done just incase an encoded data is sent
    in the body of a request made. If not, and the request body contains elements in a file
    format then the alternate consequence will be executed
    '''
    try:
        _format, str_test_img = request.data["image"].split(';base64,')

        decoded_file_test = base64.b64decode(str_test_img)
        # Returns a value if the data is encoded
        # in the right format (base64) else throws
        # an exception

        img_file = to_internal_value(request.data["image"])
    except Exception:
        if request.FILES["image"]:
            img_file = request.FILES["image"]
        else:
            return Response({'error': 'Something has gone wrong. Please check data format'}, status=status)

    img_name = default_storage.save(img_file.name, img_file)
    img_path = default_storage.path(img_name)

    # print(self.image)
    img = Image.open(img_path)
    img_array = image.img_to_array(img)
    # print(img_array)
    # print(img_array.shape)
    new_img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    dimensions = (28, 28)

    resized_shape = cv.resize(
        new_img, dimensions, interpolation=cv.INTER_AREA)
    # print(resized_shape.shape)

    # ready = np.expand_dims(resized_shape, axis=2)
    # print(ready.shape)
    ready = resized_shape.reshape(1, -1)
    # print('One row ready')
    # print(ready)
    # print(ready.shape)
    ready = ready / 255
    # print('One row scaled down to 0 and 1 by fraction of 255')
    # print(ready)
    # print(ready.shape)
    # print('One column, 784 rows reshape ongoing...')
    ready = ready.reshape(-1, 1)
    # print(ready)
    # print(ready.shape)

    try:
        # model_path = './digits/model/model.pckl'

        # path = Path(model_path)
        # print(path.exists())
        # relative_path = os.path.relpath(model_path)
        # print(relative_path)
        with open('.\models\model.pckl', 'rb') as f:
            pred = pickle.load(f)

        ans = pred.make_predictions(ready)
        num = str(ans[0])
        result = {'inference': num}
        print(f"Classified as {ans}")

        default_storage.delete(img_path)
        return Response(result, status=status.HTTP_200_OK)
        # plt.close('all')
        # plt.imshow(resized_shape, cmap='Greys')
        # plt.show()

    except:
        print('Failed to classify')
        return Response({'error': 'Something has gone wrong. Please check file type'}, status=status.HTTP_400_BAD_REQUEST)
