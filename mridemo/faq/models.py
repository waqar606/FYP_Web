from django.db import models

# Create your models here.
class faq(models.Model):
   faq_title=models.CharField(max_length=100)
   faq_desc=models.TextField()