# Generated by Django 5.0.2 on 2024-05-28 16:23

import tinymce.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Doc',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('doc_title', models.CharField(max_length=100)),
                ('doc_cat', models.CharField(max_length=200)),
                ('doc_desc', tinymce.models.HTMLField()),
                ('doc_image', models.FileField(default=None, max_length=250, null=True, upload_to='doc/')),
            ],
        ),
    ]