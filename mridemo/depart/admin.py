from django.contrib import admin
# Register your models here.
from .models import depart
class departAdmin(admin.ModelAdmin):
    list_display = ['depart_icons','depart_title','depart_desc']

admin.site.register(depart,departAdmin)
