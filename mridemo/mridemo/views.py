# views.py
'''import keras
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from PIL import Image

# Load the model
model = keras.models.load_model('clf_model_final.h5')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # Get the image from the request
        image_file = request.FILES['image']
        img = Image.open(image_file)
        
        # Resize the image
        img_resized = img.resize((224, 224))  # adjust size according to your model input

        # Convert the image to a numpy array
        img_array = np.array(img_resized)

        # If the image is grayscale (single channel), convert it to RGB
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        # If the image has an alpha channel (RGBA), remove it
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize the pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0

        # Make predictions
        predictions = model.predict(img_array)
        
        # Return the predictions
        return JsonResponse({'predictions': predictions.tolist()})
    else:
        return JsonResponse({'error': 'POST request required'})'''
import cv2
from rest_framework import status
from rest_framework.response import Response

from mridemo.utils import custom_response

"""from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
import cv2
import numpy as np
import os

# Adjust the model path to point directly to the .h5 file in the project directory
model_path = os.path.join(settings.BASE_DIR, 'clf_model_final.h5')
model = load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model input shape
    img = img / 255.0  # Rescale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input shape
    return img

def get_prediction_label(prediction):
    class_labels = ['AD', 'Normal', 'MCI']
    predicted_class_index = np.argmax(prediction)
    return class_labels[predicted_class_index]

def classification_view(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Preprocess the uploaded image
        processed_image = preprocess_image(image_path)
        
        # Make prediction using the loaded model
        prediction = model.predict(processed_image)
        
        # Get the predicted class label
        predicted_label = get_prediction_label(prediction)
        
        return render(request, 'testing.html', {'predicted_label': predicted_label, 'uploaded_file_url': uploaded_file_url})
    
    return render(request, 'testing.html')"""

from django.conf import settings
from django.shortcuts import render,redirect,HttpResponse
from depart.models import depart
from doc.models import Doc
from faq.models import faq
from contactenquiry.models import contactEnquiry
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from .middlewares import auth
import re
from django.core.files.storage import FileSystemStorage
from rest_framework.views import APIView
from PIL import Image
import os
from django.http import JsonResponse
import h5py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Define width and height
width = 200
height = 200


def load_keras_model(model_path):
    with h5py.File(model_path, 'r') as file:
        print(file.__dict__)
        model_weights_group = file['model_weights']
        model_config_group = file['model_config']

        model_weights = []
        for i in range(len(model_weights_group)):
            model_weights.append(model_weights_group['dense_1']['dense_1']['kernel:0'][()])

        model_config = model_config_group['layer_config'][()]

    model = build_model(model_config)  # You need to implement this function to build your Keras model
    model.set_weights(model_weights)
    return model


# Placeholder function to build your Keras model
def build_model(model_config):
    # Implement this function to build your Keras model using the provided model_config
    pass


def predict_result(request):
    if request.method == 'POST' and request.FILES['mri_image']:
        pass
        # mri_image = request.FILES['mri_image']
        # fs = FileSystemStorage()
        # filename = fs.save(mri_image.name, mri_image)
        # uploaded_file_url = fs.url(filename)
        #
        # # Construct the image path
        # image_path = os.path.join(settings.MEDIA_ROOT, filename)
        #
        # # Load the Keras model
        # model = load_keras_model('clf_model_final.h5')  # Adjust the path accordingly
        #
        # # Preprocess the image
        # img = preprocess_image(image_path)
        #
        # # Perform prediction using the model
        # result = model.predict(img)
        #
        # # Process result
        # # Example: Convert result to text labels like 'MCI', 'Alzheimer Detected', 'Normal'
        # result_label = 'Placeholder Result'
        #
        # return render(request, 'index.html', {
        #     'uploaded_file_url': uploaded_file_url,
        #     'result': result_label
        # })
    return render(request, 'model.html')


"""def model(request):
    return render(request,'model.html')"""


class PredictResultAPIView(APIView):
    permission_classes = []
    authentication_classes = []

    @staticmethod
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image_new = np.array(image)
        image_new = np.zeros(shape=(*image.shape[:-1], 1))
        for idx, img in enumerate(image):
            image_new[idx] = tf.image.rgb_to_grayscale(img)

        image_new = image_new.copy()
        image_new = cv2.resize(image_new, (128, 128))
        image_new = image_new / 255.0
        image_new = np.expand_dims(image_new, axis=0)
        return image_new

    @staticmethod
    def display_output(image, prediction):
        plt.imshow(image)
        plt.title(f'Prediction: {prediction}')
        plt.show()

    def post(self, request):
        try:
            print("working ................", request.FILES)
            if not request.FILES:
                return custom_response(message="No Image File Provided", success=False,
                                       response_status=status.HTTP_400_BAD_REQUEST, data={})
            mri_image = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(mri_image.name, mri_image)
            uploaded_file_url = fs.url(filename)
            print("pass")
            # Construct the image path
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            print("pass", image_path)

            model = load_model('clf_model_final.h5', compile=False)

            input_data = self.preprocess_image(image_path)
            prediction = model.predict(input_data)
            pred = np.array([np.argmax(y_) for y_ in prediction])

            result = ""
            if pred == 0:
                result = "Alzheimer Disease"
            elif pred == 1:
                result = "Mild Cognitive Impairment"
            elif pred == 2:
                result = "Cognitively Normal"

            return custom_response(response_status=status.HTTP_200_OK, message="success", data={
                "result": result,
                "uploaded_file_url": uploaded_file_url
            })
        except Exception as ex:
            raise ValueError(f"{ex}")
        



# Other Views.py
@auth
def model(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        # Preprocess the image and make predictions using your loaded model
        result = predict_image(image)
        return JsonResponse({'result': result})
    return render(request, 'model99.html')

def predict_image(image):
    # Preprocess the image (resize, normalize, etc.)
    # Use your loaded model to make predictions
    # Replace this with actual preprocessing and prediction logic
    return 'Normal'  # Example result


def about(request):
    return render(request, 'about.html')
@auth
def helpdet(request):
    faqData=faq.objects.all()
    data={
        'faqData':faqData
    }
    return render(request, 'help-questions.html',data)

def help(request):
    return render(request, 'help.html')

@auth
def home(request):
    return render(request, 'index-7.html')

def signup(request):
    if request.method=="POST":
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')
        if not re.match("^[A-Za-z]+$", uname):
            return render(request, 'signup.html', {
                'error': 'Username should contain only letters',
                'username': uname,
                'email': email
            })

        

        # Password match validation
        if pass1 != pass2:
            return render(request, 'signup.html', {
                'error': 'Passwords do not match',
                'username': uname,
                'email': email
            })

        # Create the user
        my_user=User.objects.create_user(uname,email,pass1)
        my_user.save()
        return redirect('login')
    return render(request, 'signup.html')



def log(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        
        else:
            return render(request, 'login.html', {
                'error': 'Invalid username or password',
                'username': username,
            })
    return render(request, 'login.html')


def LogOut(request):
    logout(request)
    return redirect('login')


def reviews(request):
    return render(request, 'reviews.html')

def treatment(request):
    return render(request, 'treatments.html')

# def redirect(request):
#     return render(request, 'Redirect.html')

def membership(request):
    return render(request, 'membership.html')

def working(request):
    return render(request, 'how-it-works.html')
@auth
def contact(request):
    return render(request, 'contact-us.html')

def saveEnquiry(request):
    if request.method=="POST":
        name=request.POST.get('name')
        subject=request.POST.get('subject')
        email=request.POST.get('email')
        relationship=request.POST.get('relationship')
        message=request.POST.get('message')
        en=contactEnquiry(name=name,subject=subject,email=email,relationship=relationship,message=message)
        en.save()       
    return render(request, 'contact-us.html')

@auth
def doctors(request):
    docData=Doc.objects.all()
    mydata={
        'docData':docData
    }
    return render(request, 'doctors.html',mydata)

def doctorsdet(request):
    return render(request, 'doctors-detailed.html')
@auth
def department(request):
    departData=depart.objects.all()
    data={
       'departData':departData
    }
    return render(request, 'departments-2.html',data)

def departmentdet(request):
    return render(request, 'departments-detailed.html')

def blog(request):
    return render(request, 'blog-masonry.html')

def blogdet(request):
    return render(request, 'blog-detailed.html')

def app1(request):
    return render(request, 'appointment-step1.html')

def app2(request):
    return render(request, 'appointment-step2.html')

def app3(request):
    return render(request, 'appointment-step3.html')

def app4(request):
    return render(request, 'appointment-step4.html')


