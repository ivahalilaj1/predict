# Generated by Django 2.0.7 on 2018-09-10 22:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('category', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CategoricalCategory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, verbose_name='Name')),
                ('coefficient', models.FloatField(verbose_name='Coefficient')),
            ],
        ),
        migrations.CreateModel(
            name='CategoricalRegressionVariable',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(max_length=255)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='ContinuousRegressionVariable',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(max_length=255)),
                ('coefficient', models.FloatField(verbose_name='Coefficient')),
                ('min', models.IntegerField(verbose_name='Min')),
                ('max', models.IntegerField(verbose_name='Max')),
                ('step', models.FloatField(verbose_name='Step')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='LogisticRegression',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
        migrations.CreateModel(
            name='PredictModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254)),
                ('submitter', models.CharField(max_length=255)),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(verbose_name='Description')),
                ('model_type', models.CharField(choices=[('lr', 'Logistic regression'), ('cr', 'Cox regression')], max_length=100)),
                ('constant', models.FloatField(blank=True, null=True, verbose_name='Logistic Regression constant')),
                ('categories', models.ManyToManyField(help_text='Categorize this item.', related_name='predict_models', to='category.Category')),
                ('tags', models.ManyToManyField(help_text='Tag this item.', related_name='predict_models', to='category.Tag')),
            ],
        ),
        migrations.AddField(
            model_name='continuousregressionvariable',
            name='predict_model',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='continuous_regression_variables', to='predict.PredictModel'),
        ),
        migrations.AddField(
            model_name='categoricalregressionvariable',
            name='predict_model',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='categorical_regression_variables', to='predict.PredictModel'),
        ),
        migrations.AddField(
            model_name='categoricalcategory',
            name='regression_variable',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='categorical_categories', to='predict.CategoricalRegressionVariable'),
        ),
    ]
