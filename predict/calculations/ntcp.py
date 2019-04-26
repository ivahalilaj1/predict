import operator
from functools import reduce
from math import log, exp


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def ntcp_calculation(self, data):
    # TODO: fix circular import
    from predict.models import CategoricalCategory

    continuous_vars = self.continuous_regression_variables.all()
    categorical_vars = self.categorical_regression_variables.all()
    labels = [var.label for var in continuous_vars] + [var.label for var in categorical_vars]
    continuous_map = {variable.label: variable
                      for variable in continuous_vars}
    categorical_map = {variable.label: variable
                       for variable in categorical_vars}
    categorical_coeff = []
    d_mean = None
    for field, value in data.items():
        if field not in labels:
            continue
        if field in categorical_map:
            category = CategoricalCategory.objects.get(pk=value)
            categorical_coeff.append((category.coefficient, 1))
        elif field in continuous_map:
            d_mean = continuous_map[field].coefficient * float(value)
    y_risk_50_0 = 1.19
    or_variables = prod([coeff[0] * coeff[1] for coeff in categorical_coeff])
    y_risk_50 = y_risk_50_0 - (0.25 * log(or_variables))
    d_risk_50_0 = 34.4
    d_risk_50 = d_risk_50_0 * (1 - (log(or_variables) / (4 * y_risk_50_0)))
    p = 1 / (1 + exp(4 * y_risk_50 * (1 - (d_mean / d_risk_50))))
    return p * 100
