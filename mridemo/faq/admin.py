from django.contrib import admin
# Register your models here.
from .models import faq
class FAQAdmin(admin.ModelAdmin):
    list_display = ['faq_title','faq_desc']
admin.site.register(faq,FAQAdmin)