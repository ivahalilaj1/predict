# Generated by Django 2.0.7 on 2018-10-20 19:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0008_auto_20181003_2103'),
    ]

    operations = [
        migrations.AddField(
            model_name='continuousregressionvariable',
            name='scale_center',
            field=models.FloatField(blank=True, null=True, verbose_name='Scaling factor'),
        ),
        migrations.AddField(
            model_name='continuousregressionvariable',
            name='scale_factor',
            field=models.FloatField(default=1.0, verbose_name='Scaling factor'),
        ),
    ]
