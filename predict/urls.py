from django.urls import path
from django.conf.urls import url
from predict import views
from .views import sergeymodel, disclaimer, select_image
app_name = 'predict'

urlpatterns = [
    path('list/', views.ListPredictModelView.as_view(), name='list'),
    path('new/', views.CreatePredictModelView.as_view(), name='new'),
    path('<int:pk>/', views.PredictModelView.as_view(), name='model'),
    url(r'^$', views.IndexView.as_view(), name='index'),
    path('sergeymodel', sergeymodel, name='sergeymodel'),
    path('disclaimer', disclaimer, name='disclaimer'),
    path('select_image', select_image, name='select_image'),
    url(r'^predictmodel/entry/$',views.PredictModelEntry.as_view(),name='predictmodel-entry'),
    url(r'^predictmodel/(?P<pk>[0-9]+)/$', views.PredictModelUpdate.as_view(), name='predictmodel-update'),
    url(r'^predictmodel/(?P<pk>[0-9]+)/delete$', views.PredictModelDelete.as_view(), name='predictmodel-delete'),
]
