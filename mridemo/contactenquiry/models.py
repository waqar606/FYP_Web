from django.db import models
# Create your models here.
class contactEnquiry (models.Model):
    name=models.CharField(max_length=50)
    subject=models.CharField(max_length=100)
    email=models.CharField(max_length=60)
    relationship=models.CharField(max_length=50)
    message=models.TextField()