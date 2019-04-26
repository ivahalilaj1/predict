from __future__ import unicode_literals
from django import template
from django.db import models
from django.template import Context
from django.utils.text import slugify
from django.utils.translation import ugettext_lazy as _

from predict.calculations.logistic_regression import lr_calculation
from predict.calculations.ntcp import ntcp_calculation
from predict.calculations.linear_regression import lrm_calculation

from django.urls import reverse


    
class LogisticRegression(models.Model):

    @property
    def regression_variables(self):
        return self.regression_variables.all()


class ScoreTemplate(models.Model):
    name = models.CharField(max_length=100)
    template = models.TextField()

    def get_rendered_template(self, context):
        return template.Template(self.template).render(Context(context))

    def __str__(self):
        return self.name


class PredictModel(models.Model):
    LOGISTIC_REGRESSION = 'lrm'
    COX_REGRESSION = 'cr'
    NTCP = 'ntcp'
    LINEAR_REGRESSION = 'lm'
    SEGMENTATION = 'seg'

    MODEL_TYPES = (
        (LOGISTIC_REGRESSION, _('Logistic regression')),
        (LINEAR_REGRESSION, _('Linear regression')),
        (COX_REGRESSION, _('Cox regression')),
        (NTCP, _('NTCP model')),
        (SEGMENTATION, _('Segmentation model')),
    )

    email = models.EmailField()
    submitter = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    description = models.TextField(_('Description'))
    references = models.TextField(_('References'), blank=True, null=True)
    model_type = models.CharField(choices=MODEL_TYPES, max_length=100)
    constant = models.FloatField(_('Logistic Regression constant'), blank=True, null=True)
    score_template = models.ForeignKey(ScoreTemplate, blank=True, null=True, on_delete=models.SET_NULL)

    is_approved = models.BooleanField(default=False)
    def get_absolute_url(self):
        return reverse('predict:index')
        PredictModel.objects.filter(is_approved = True)

    categories = models.ManyToManyField(
        'category.Category',
        help_text='Categorize this item.',
        related_name='predict_models',
    )
    tags = models.ManyToManyField(
        'category.Tag',
        help_text='Tag this item.',
        related_name='predict_models',
    )

    def __str__(self):
        return f'{self.model_type}: {self.name}'

    @property
    def slug(self):
        return slugify(self.name)

    def calculate(self, data):
        if self.model_type == self.NTCP:
            return ntcp_calculation(self, data)
        if self.model_type == self.LOGISTIC_REGRESSION:
            return lr_calculation(self, data)
        if self.model_type == self.LINEAR_REGRESSION:
            return lrm_calculation(self, data)
        else:
            raise ValueError('calculate not implemented for this model')



class RegressionVariable(models.Model):
    label = models.CharField(max_length=255)

    class Meta:
        abstract = True


class ContinuousRegressionVariable(RegressionVariable):
    predict_model = models.ForeignKey(
        PredictModel, related_name='continuous_regression_variables',
        on_delete=models.CASCADE)
    coefficient = models.FloatField(_('Coefficient'))
    min = models.FloatField(_('Min'))
    max = models.FloatField(_('Max'))
    step = models.FloatField(_('Step'))
    scale_factor = models.FloatField(_('Scaling factor'), default=1.0, )
    scale_center = models.FloatField(_('Coeffient centre'), blank=True, null=True)

    def __str__(self):
        return f'{self.label} ({self.min}, {self.max}, {self.step})'


class CategoricalRegressionVariable(RegressionVariable):
    predict_model = models.ForeignKey(
        PredictModel, related_name='categorical_regression_variables',
        on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.label}'


class CategoricalCategory(models.Model):
    regression_variable = models.ForeignKey(
        CategoricalRegressionVariable, on_delete=models.CASCADE,
        related_name='categorical_categories')

    name = models.CharField(_('Name'), max_length=255)
    coefficient = models.FloatField(_('Coefficient'))



