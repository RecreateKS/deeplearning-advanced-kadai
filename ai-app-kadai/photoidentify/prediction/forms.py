from django import forms

#画像をアップロード
class ImageUploadForm(forms.Form):
    image = forms.ImageField()

