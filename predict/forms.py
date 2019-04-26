from django.forms import ModelForm
from django.forms.models import inlineformset_factory
from django import forms

from predict.models import (
    ContinuousRegressionVariable, CategoricalRegressionVariable, PredictModel)

class UploadFileForm(forms.Form):
    file = forms.FileField()

class SelectFileForm(forms.Form):
    image = forms.FileField()

class PredictModelForm(ModelForm):
    class Meta:
        model = PredictModel
        fields = '__all__'


ContinuousRegressionVariableFormSet = inlineformset_factory(
    PredictModel, ContinuousRegressionVariable, fields='__all__', extra=1, can_delete=False)
CategoricalRegressionVariableFormSet = inlineformset_factory(
    PredictModel, CategoricalRegressionVariable, fields='__all__', extra=1, can_delete=False)

class PredictModelFormUser(forms.ModelForm):
    class Meta:
        model = PredictModel
        fields = '__all__'
