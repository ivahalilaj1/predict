from django.views.generic import TemplateView


class IndexView(TemplateView):

    template_name = 'index.html'


class LegalView(TemplateView):

    template_name = 'legal.html'
