from django.contrib import admin
from .models import Doc
# Register your models here.
class DocAdmin(admin.ModelAdmin):
    list_display = ['doc_title','doc_cat','doc_desc','doc_image']
admin.site.register(Doc,DocAdmin)