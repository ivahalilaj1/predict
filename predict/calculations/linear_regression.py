def lrm_calculation(predict_model, data):
    from predict.models import CategoricalCategory

    continuous_vars = predict_model.continuous_regression_variables.all()
    categorical_vars = predict_model.categorical_regression_variables.all()
    labels = [var.label for var in continuous_vars] + [var.label for var in categorical_vars]
    continuous_map = {variable.label: variable
                      for variable in continuous_vars}
    categorical_map = {variable.label: variable
                       for variable in categorical_vars}
    value_coeff = []
    for field, value in data.items():
        if field not in labels:
            continue
        if field in categorical_map:
            category = CategoricalCategory.objects.get(pk=value)
            value_coeff.append((category.coefficient, 1))
        elif field in continuous_map:
            value_coeff.append(((continuous_map[field].coefficient - continuous_map[field].scale_center) /
                                continuous_map[field].scale_factor,
                                float(value)))

    return predict_model.constant + sum([coeff[0] * coeff[1] for coeff in value_coeff])
