from __future__ import print_function
import traceback
import sys,os
from django.views import generic
from category.models import Category
from django.http import HttpResponseRedirect
from django.views.generic import FormView
from extra_views import CreateWithInlinesView, InlineFormSet
from predict.form_generator import form_from_predict_model
from predict.forms import PredictModelForm, \
    CategoricalRegressionVariableFormSet, ContinuousRegressionVariableFormSet
from django.core.files.storage import default_storage
from predict.models import (
    CategoricalRegressionVariable, ContinuousRegressionVariable,
    PredictModel)
from django.shortcuts import render, get_object_or_404
from .forms import UploadFileForm
from .lung_model import Get_segmentation
from django.views.generic import (TemplateView,ListView,
                                  DetailView,CreateView,
                                  UpdateView,DeleteView)
from django.urls import reverse_lazy
import numpy as np
import SimpleITK as sitk

class ContinuousRegressionVariableInline(InlineFormSet):
    model = ContinuousRegressionVariable
    fields = '__all__'

class CategoricalRegressionVariableInline(InlineFormSet):
    model = CategoricalRegressionVariable
    fields = '__all__'


class CreatePredictModelView(CreateWithInlinesView):
    model = PredictModel
    inlines = [CategoricalRegressionVariableInline,
               ContinuousRegressionVariableInline]
    fields = '__all__'


class PredictModelView(FormView):
    """
    View to calculate with the parameters of a  model
    """
    template_name = 'predict/predict_model.html'

    def get_object(self):
        return get_object_or_404(PredictModel, pk=self.kwargs['pk'])

    def get_form_class(self):
        predict_model = self.get_object()
        return form_from_predict_model(predict_model)

    def get_initial(self):
        initial = super().get_initial()
        initial['predict_model'] = self.kwargs['pk']
        return initial

    def get_context_data(self, **kwargs):
        context_data = super().get_context_data(**kwargs)
        context_data.update({
            'predict_model': self.get_object(),
        })
        return context_data

    def form_valid(self, form):
        predict_model = self.get_object()
        try:
            model_score = predict_model.calculate(form.data)
        except ValueError:
            model_score = 'Error during calculation'
        score_output = predict_model.score_template.get_rendered_template({'model_score': model_score})
        context = self.get_context_data(form=form)
        context.update({'model_score': model_score, 'score_output': score_output})
        return self.render_to_response(context)


class PredictModelCreateView(CreateView):
    template_name = 'predict/predict_model_add.html'
    model = PredictModel
    form_class = PredictModelForm
    success_url = 'success/'

    def get(self, request, *args, **kwargs):
        """
        Handles GET requests and instantiates blank versions of the form
        and its inline formsets.
        """
        self.object = None
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        categorical_variables_form = CategoricalRegressionVariableFormSet()
        continuous_variables_form = ContinuousRegressionVariableFormSet()
        return self.render_to_response(
            self.get_context_data(form=form,
                                  categorical_variables_form=categorical_variables_form,
                                  continuous_variables_form=continuous_variables_form))

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests, instantiating a form instance and its inline
        formsets with the passed POST variables and then checking them for
        validity.
        """
        self.object = None
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        categorical_variables_form = CategoricalRegressionVariableFormSet(self.request.POST)
        continuous_variables_form = ContinuousRegressionVariableFormSet(self.request.POST)
        if (form.is_valid() and continuous_variables_form.is_valid() and
            categorical_variables_form.is_valid()):
            return self.form_valid(form, categorical_variables_form,
                                   continuous_variables_form)
        else:
            return self.form_invalid(form, categorical_variables_form, continuous_variables_form)

    def form_valid(self, form, categorical_variables_form,
                   continuous_variables_form):
        """
        Called if all forms are valid. Creates a Predict model instance along
        with associated variables and redirects to success page.
        """
        self.object = form.save()
        categorical_variables_form.instance = self.object
        categorical_variables_form.save()
        continuous_variables_form.instance = self.object
        continuous_variables_form.save()
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form, categorical_variables_form,
                     continuous_variables_form):
        """
        Called if a form is invalid. Re-renders the context data with the
        data-filled forms and errors.
        """
        return self.render_to_response(
            self.get_context_data(
                form=form,
                categorical_variables_form=categorical_variables_form,
                continuous_variables_form=continuous_variables_form))

class ListPredictModelView(TemplateView):
    template_name = 'predict/list.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        categories = Category.objects.filter(parent=None).order_by('title')
        context.update({
            'categories': categories,
            'model_count': PredictModel.objects.count(),
        })
        return context

class PredictModelFormUser(FormView):
     model = PredictModel
class IndexView(generic.ListView):
    context_object_name = 'predictmodel_list'
    template_name = 'predict/index.html'

    def get_queryset(self):
        return PredictModel.objects.filter(is_approved = True)

class PredictModelEntry(CreateView):
    model = PredictModel
    fields = ['email', 'submitter', 'name', 'description', 'references', 'model_type', 'constant', 'score_template', 'categories','tags']


class PredictModelUpdate(UpdateView):
    model = PredictModel
    fields = ['email', 'submitter', 'name', 'description', 'references', 'model_type', 'constant', 'score_template', 'categories','tags']

class PredictModelDelete(DeleteView):
    model = PredictModel
    success_url = reverse_lazy('predict:index')


class ScoreTemplateUpdate(UpdateView):
    model = PredictModel
    fields = ['name', 'template']

class ScoreTemplateEntry(CreateView):
    model = PredictModel
    fields = ['name', 'template']

class ScoreTemplateDelete(DeleteView):
    model = PredictModel
    success_url = reverse_lazy('predict:index')

class RegressionVariableEntry(CreateView):
    model = PredictModel
    fields = ['label']
class ContinuousRegressionVariableEntry(CreateView):
    model = PredictModel
    fields = ['coefficient', 'min', 'max', 'step', 'scale_factor', 'scale_center']

class CategoricalCategoryEntry(CreateView):
    model = PredictModel
    fields = ['name', 'coefficient']


def sergeymodel(request):

    form = UploadFileForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        filename = "tempimage.dcm" # received file name
        file_obj = request.FILES['myfile']
        with default_storage.open('tmp/'+filename, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)

        imagefile = sitk.ReadImage(os.path.join('./media/tmp',filename))  #('tmp/'+filename)

        temp_image = np.squeeze(np.array(sitk.GetArrayFromImage(imagefile),np.float))

        jpeg_path = Get_segmentation(temp_image)
        return render(request, 'sergeymodel.html', {'form': form, 'path':jpeg_path})

    return render(request, 'sergeymodel.html', {'form': form, 'path':None})

def disclaimer(request):
    form = UploadFileForm
    return render(request, 'disclaimer.html')
def select_image(request):
    form = UploadFileForm
    return render(request,'select_image.html')
