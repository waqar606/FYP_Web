"""
URL configuration for mridemo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

"""from secondApp import views
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    #path('', views.model,name="model"),#
    path('', include('secondApp.urls')),  # Include the URLs configuration of secondApp
]"""
# your_project/urls.py

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
from .views import predict_result, PredictResultAPIView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', predict_result, name='predict_result'),
    path('api/predict/', PredictResultAPIView.as_view()),
    path('model/',views.model,name='model'),
    path('about/',views.about,name='about'),
    path('help/',views.help,name='help'),
    path('helpdet/',views.helpdet,name='helpdet'),
    path('home/',views.home,name='home'),
    path('login/',views.log,name='login'),
    path('logout/',views.LogOut,name='logout'),
    path('signup/',views.signup,name='sigup'),
    path('reviews/',views.reviews,name='review'),
    path('treatment/',views.treatment,name='treatment'),
    path('redirect/',views.redirect,name='redirect'),
    path('membership/',views.membership,name='membership'),
    path('working/',views.working,name='working'),
    path('contact/',views.contact,name='contact'),
    path('saveenquiry/',views.saveEnquiry,name='saveenquiry'),
    path('doctor/',views.doctors,name='doctor'),
    path('doctordet/',views.doctorsdet,name='doctordet'),
    path('department/',views.department,name='dep'),
    path('departmentdet/',views.departmentdet,name='depdet'),
    path('blog/',views.blog,name='blog'),
    path('blogdet/',views.blogdet,name='blogdet'),
    path('app1/',views.app1,name='app1'),
    path('app2/',views.app2,name='app2'),
    path('app3/',views.app3,name='app3'),
    path('app4/',views.app4,name='app4'),
    #path('', views.classification_view, name='classification'),
    #path('',views.test),
    #path('predict/',views.predict, name='predict')
    #path('', include('secondApp.urls')),  # Include the URLs configuration of secondApp
    # Other URL patterns for your project...
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)