from django import forms


def form_from_predict_model(predict_model):
    fields = {
        'predict_model': forms.IntegerField(widget=forms.HiddenInput())
    }

    # TODO: assert labels are unique, maybe add pk to label name?!!!
    continuous_vars = predict_model.continuous_regression_variables.all()
    categorical_vars = predict_model.categorical_regression_variables.all()
    for variable in continuous_vars:
        # min / max
        fields[variable.label] = forms.FloatField(
            label=variable.label,
            min_value=variable.min,
            max_value=variable.max,
            widget=forms.NumberInput(attrs={'step': str(variable.step)}))
    for variable in categorical_vars:
        # choices
        categories = variable.categorical_categories.all()
        fields[variable.label] = forms.ChoiceField(
            label=variable.label,
            choices=[(category.id, category.name)
                     for category in categories]
        )
    return type('{}Form'.format(predict_model.slug), (forms.Form,), fields)
