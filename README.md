# Pumpwood Models

This package facilitates development of new models for Pumpwood structure. All
models are implemented as a class with same functions so modeling matrix and

<p align="center" width="60%">
  <img src="doc/sitelogo-horizontal.png" /> <br>

  <a href="https://en.wikipedia.org/wiki/Cecropia">
    Pumpwood is a native brasilian tree
  </a> which has a symbiotic relation with ants (Murabei)
</p>

# How use it?
It is necessary to override some methods, to configure models to be used on
Pumpwood:

## Get model parameters
`get_model_parameters(cls, partial_parameters: dict = {})`

This function will return the model parameters available for model definition, it receives a partial parameter argument `partial_parameter`. Using this argument it is possible to configure parameter restriction when some parameters are filled.

One example of this is when creating a GLM models, some link families may have
some parameters that others does not have.

It must return a dictionary with:
- **required:** A dictionary with description of the required parameters for the
  model.
- **optional:** A dictionary with description of the optional parameters for the
  model.

## Run model
```
run_model(cls, data: pd.DataFrame, columns_info: dict,
          output_var_attribute_id: int, model_parameter: dict):
```
It have as parameters the estimation matrix `data`, a dictionary with
information of the columns, an `output_var_attribute_id` indicating
which `attribute_id` is considered as output of the model and
a dictionary with `model_parameters` that can be used to configure
the algorithm.

It must return a dictionary with:
- **model [object]:** A python object that can be pickled with the estimated model.
- **residuals [pd.DataFrame]:** A dataframe with the residuals of the intra-amostral
  estimation. It must be returned as a pandas dataframe with columns:
  - time [datetime]: Time associated with the results.
  - modeling_unit_id [int]: Modeling unit associated with the results.
  - geoarea_id [int]: Geoarea associated with the results.
  - var_attribute_id [int]: Attribute associated with the output of the model.
  - type [str]: A type of the residual that is returned, it must have
    max 20 characters. Ex.: fitted, residual_desviance.
- **parameters_simple [pd.DataFrame]:** Simple parameters to be saved on
  database. Columns:
  - name [str[30] or None]: Name of the parameter.
  - type [str[30] or None]: Type of the parameter. Some examples: ``[exogenous_parameter, intercept, converged, model_selection]``.
  - time [pd.Timestamp or None]: If parameter is time variable, this will
    be associated with reference time.
  - modeling_unit_id (int or None): Modeling unit associated with parameter.
  - geoarea_id (int or None): Geoarea associated with parameter.
  - var_attribute_id (int or None): Input attribute (DescriptionAttribute)
      associated with parameter.
  - var_geoattribute_id (int or None): Input geo-attribute
      (DescriptionGeoAttribute) associated with parameter.
  - var_calendar_id (int or None): Input calendar
      (DescriptionCalendar) associated with parameter.
  - var_dummy_id (int or None): Input dummy variable
      (DescriptionDummy) associated with parameter.
  - value (dict): A dictionary with parameter results.
- **parameters_complex [list[dict]]:** A list of dictionary that will be saved
  as a complex parameter results, each list entry must contain:
    - name [str[50]]: Name of the complex parameter
    - data [dict]: Complex result to be saved as a dictionary on database.
    - file [File Stream or None]: A file object or Python io.BytesIO with
      seek(0) with a binary result.


# Example
```python
"""Pumpwood GLM model."""
import io
import copy
import scipy
import statsmodels
import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
from pumpwood_communication.microservices import PumpWoodMicroService
from pumpwood_communication.serializers import pumpJsonDump
from pumpwood_models.model import PumpwoodModel


class Model(PumpwoodModel):
    """Model runner for StatsModels Nice Models."""

    model_type = "statsmodels-nicemodel"

    family__dict = {
        'binomial': statsmodels.genmod.families.family.Binomial,
        'gamma': statsmodels.genmod.families.family.Gamma,
        'gaussian': statsmodels.genmod.families.family.Gaussian,
        'inversegaussian': statsmodels.genmod.families.family.InverseGaussian,
        'negativebinomial':
            statsmodels.genmod.families.family.NegativeBinomial,
        'poisson': statsmodels.genmod.families.family.Poisson,
        'tweedie': statsmodels.genmod.families.family.Tweedie
    }

    link_dict = {
        None: None,
        'cdflink': statsmodels.genmod.families.links.CDFLink,
        'cloglog': statsmodels.genmod.families.links.CLogLog,
        'log': statsmodels.genmod.families.links.Log,
        'logit': statsmodels.genmod.families.links.Logit,
        'negativebinomial': statsmodels.genmod.families.links.NegativeBinomial,
        'power': statsmodels.genmod.families.links.Power,
        'cauchy': statsmodels.genmod.families.links.cauchy,
        'identity': statsmodels.genmod.families.links.identity,
        'inverse_power': statsmodels.genmod.families.links.inverse_power,
        'inverse_squared': statsmodels.genmod.families.links.inverse_squared,
        'nbinom': statsmodels.genmod.families.links.nbinom,
        'probit': statsmodels.genmod.families.links.probit,
    }
    temp_dir = None

    @classmethod
    def get_model_parameters(cls, partial_parameters: dict = {}):
        parameters_required = {}
        parameters_optional = {}
        parameters_required['family'] = {
            'type': 'options',
            'edit': True,
            'nullable': False,
            'description': 'family of the glm estimation',
            'in': [
                {'value': 'binomial', 'description': 'binomial'},
                {'value': 'gamma', 'description': 'gamma'},
                {'value': 'gaussian', 'description': 'gaussian'},
                {'value': 'inversegaussian',
                 'description': 'inverse gaussian'},
                {'value': 'negativebinomial',
                 'description': 'negative binomial'},
                {'value': 'poisson', 'description': 'poisson'},
                {'value': 'tweedie', 'description': 'tweedie'}
            ]
        }

        parameters_required['intercept'] = {
            'type': 'bool',
            'edit': True,
            'nullable': False,
            'description': 'add an intercept to model',
            'default': True,
        }

        family = partial_parameters.get("family")
        parameters_optional['link'] = {
            'type': 'options',
            'edit': True,
            'nullable': True,
            'description': 'link function to be used in family',
            'in': []}
        if family == "binomial":
            parameters_optional['link']["in"] = [
                {'value': None, 'description': ''},
                {'value': 'logit', 'description': 'logit'},
                {'value': 'probit', 'description': 'probit'},
                {'value': 'cauchy', 'description': 'cauchy'},
                {'value': 'log', 'description': 'log'},
                {'value': 'cloglog', 'description': 'cloglog'},
                {'value': 'identity', 'description': 'identity'}
            ]

    @classmethod
    def run_model(cls, data: pd.DataFrame, columns_info: dict,
                  output_var_attribute_id: int, model_parameter: dict):
        # [run model estimation stuff here...]
        return {
          "model": fitted_model_obj: object,
          "residuals": residuals_df: pd.DataFrame,
          "parameters_simple": parameters_simple_df: pd.DataFrame,
          "parameters_complex": parameters_complex: list,
        }
```
