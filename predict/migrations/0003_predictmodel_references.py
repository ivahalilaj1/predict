# Generated by Django 2.0.7 on 2018-10-01 20:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0002_auto_20180930_1713'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictmodel',
            name='references',
            field=models.TextField(default='', verbose_name='References'),
            preserve_default=False,
        ),
    ]
