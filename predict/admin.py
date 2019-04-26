from django.contrib import admin
from nested_admin.nested import (
    NestedModelAdmin, NestedInlineModelAdmin, NestedTabularInline)

from predict.models import (
    PredictModel, ContinuousRegressionVariable,
    CategoricalRegressionVariable, CategoricalCategory, ScoreTemplate)


class ContinuousRegressionVariableInline(NestedTabularInline):
    model = ContinuousRegressionVariable


class CategoricalRegressionVariableInline(NestedTabularInline):
    model = CategoricalRegressionVariable


class CategoricalCategoryInline(NestedTabularInline):
    model = CategoricalCategory


class ContinuousRegressionVariableAdmin(NestedInlineModelAdmin):
    model = ContinuousRegressionVariable
    sortable_field_name = 'label'


class CategoricalRegressionVariableAdmin(NestedInlineModelAdmin):
    inlines = [
        CategoricalCategoryInline
    ]
    model = CategoricalRegressionVariable
    sortable_field_name = 'label'


@admin.register(PredictModel)
class PredictModelAdmin(NestedModelAdmin):
    inlines = [
        CategoricalRegressionVariableInline,
        ContinuousRegressionVariableInline,
    ]


@admin.register(CategoricalRegressionVariable)
class CategoricalRegressionVariableAdmin(NestedModelAdmin):
    inlines = [
        CategoricalCategoryInline
    ]


@admin.register(ScoreTemplate)
class ScoreTemplateAdmin(admin.ModelAdmin):
    pass


