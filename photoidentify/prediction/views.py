from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os

# Create your views here.
def predict(request):

    # GETリクエストによるアクセス時の処理
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    # POSTリクエストによるアクセス時の処理
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            #array形式に変換
            img_array = img_to_array(img)
            #形状を「任意のサンプル数×224×224×3」に変換
            img_array = img_array.reshape((1, 224, 224, 3))
            #前処理
            img_array = preprocess_input(img_array)
            #予測モデルのパス
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            #予測モデルの読み込み
            model = load_model(model_path)
            #各カテゴリに属する確率
            result = model.predict(img_array)
            #上位の予測カテゴリと確率を抽出
            prediction = decode_predictions(result)[0]
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
        