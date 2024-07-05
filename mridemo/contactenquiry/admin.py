from django.contrib import admin
from .models import contactEnquiry
# Register your models here.
class ContactAdmin(admin.ModelAdmin):
    list_display=['name','subject','email','relationship','message']

admin.site.register(contactEnquiry,ContactAdmin)