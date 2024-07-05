from django.db import models
from tinymce.models import HTMLField
# Create your models here.
class Doc(models.Model):
    doc_title=models.CharField(max_length=100)
    doc_cat=models.CharField(max_length=200)
    doc_desc=HTMLField()
    doc_image=models.FileField(upload_to="doc/",max_length=250,null=True,default=None)