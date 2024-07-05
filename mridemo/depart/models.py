# models.py in your app directory
from django.db import models
class depart(models.Model):
   depart_icons=models.CharField(max_length=50)
   depart_title=models.CharField(max_length=50)
   depart_desc=models.TextField()
   
# Create your models here.
