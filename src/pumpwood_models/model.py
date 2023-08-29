"""Model runner module."""
import os
import io
import pandas as pd
import pickle
import datetime
import shutil
import ntpath
from sqlalchemy import create_engine
from abc import ABC, abstractmethod
from typing import List
from pumpwood_communication.microservices import PumpWoodMicroService
from pumpwood_miscellaneous.storage import PumpWoodStorage
from pumpwood_models.queries import rawdata_summary_query


class PumpwoodModel(ABC):
    """Model runner for StatsModels GLMs."""

    model_type = None
    "Define the model type, this name will be avaiable at Model Description."

    def __init__(self, microservice: PumpWoodMicroService,
                 storage_object: PumpWoodStorage):
        """Create a new model object.

        Args:
            microservice (PumpWoodMicroService): Microservice object to connect
                with PumpWood End-Points.
            storage_object (PumpWoodStorage): A Pumpwood storage object.
        Raise:
            AttributeError("model_type is None, it must be set as class
                property."): This raise will occour when the model_type is note
                set on sub-class implementation.
        """
        if self.model_type is None:
            raise AttributeError(
                "model_type is None, it must be set as class property.")

        self._microservice = microservice
        self._microservice.login()
        self._storage_object = storage_object
        self.temp_dir = None

    def __del__(self):
        """__del__."""
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir)

    @classmethod
    def save_temp_file(cls, content, suflix: str, extention: str,
                       binary: bool = False):
        """
        Save a temporary file.

        Args:
            content (text or bytes): Content of the file to be saved.
            suflix (str): Sufix of the file to be saved.
            extention (str): File extension.
            binary ():

        """
        template = "temp/{time}_model_queue__{suflix}.{extention}"
        path = template.format(
            time=datetime.datetime.utcnow().isoformat(),
            suflix=suflix, extention=extention)
        type_open = "wb" if binary else "w"
        with open(path, type_open) as file:
            file.write(content)
        return open(path, "rb")

    @classmethod
    def load_temp_file(cls, content, suflix: str, extention: str,
                       binary: bool = False):
        """Save a temporary file."""
        template = "temp/{time}_model_queue__{suflix}.{extention}"
        path = template.format(
            time=datetime.datetime.utcnow().isoformat(),
            suflix=suflix, extention=extention)
        os.path.isfile(path)
        return open(path, "rb")

    def set_queue_error(self, msg: dict, error_msg):
        model_queue_obj = self._microservice.retrieve(
            model_class="ModelQueue", pk=msg['pk'])
        model_queue_obj['status'] = 'error_estimation'
        model_queue_obj['msg'] = error_msg
        return self._microservice.save(obj_dict=model_queue_obj)

    def process_estimation_queue(self, model_queue: dict):
        """Process message queue."""
        if not os.path.exists("temp/"):
            os.makedirs("temp/")
        else:
            shutil.rmtree("temp/")
            os.makedirs("temp/")

        start_estimation = datetime.datetime.utcnow()
        self._microservice.login()

        model_queue = self._microservice.retrieve(
                model_class="ModelQueue", pk=model_queue['pk'])
        model_queue['status'] = 'running_estimation'
        model_queue['start_estimation'] = start_estimation
        model_queue = self._microservice.save(obj_dict=model_queue)

        model = self._microservice.list_one(
            model_class="DescriptionModel", pk=model_queue['model_id'])

        print('Geting model parameters...')
        estimation_data = self.get_estimation_data(model_queue=model_queue)
        columns_info = model_queue['model_matrix_columns']
        output_var_attribute_id = model["output_var_attribute_id"]
        model_parameter = model['parameters']

        print('Running model...')
        model_results = self.run_model(
            data=estimation_data, columns_info=columns_info,
            output_var_attribute_id=output_var_attribute_id,
            model_parameter=model_parameter)

        print('# Saving results')
        print('## Saving residuals...')
        self.save_residuals(
            model_queue=model_queue,
            residuals=model_results['residuals'])

        print('## Saving parameters_simple...')
        self.save_parameters_simple(
            model_queue=model_queue,
            parameters_simple=model_results['parameters_simple'])

        print('## Saving parameters_complex...')
        self.save_parameters_complex(
            model_queue=model_queue,
            parameters_complex=model_results['parameters_complex'])

        print('## Saving model file...')
        final_result = self.save_model_file(
            model_queue=model_queue,
            model_fit=model_results['model'])
        print('@Finished!!')
        shutil.rmtree("temp/")
        return final_result

    def process_predicton_queue(self, prediction_queue: dict) -> dict:
        """
        Process message queue.

        Args:
            prediction_queue [dict]: Serialized prediction queue.
        Return [dict]:
            Return prediction queue serialized object.
        """
        if not os.path.exists("temp/"):
            os.makedirs("temp/")
        else:
            shutil.rmtree("temp/")
            os.makedirs("temp/")

        self._microservice.login()
        prediction_queue = self._microservice.retrieve(
                model_class="PredictionQueue", pk=prediction_queue['pk'])
        prediction_queue['status'] = 'running_prediction'
        prediction_queue['start_prediction'] = datetime.datetime.utcnow()
        prediction_queue = self._microservice.save(
            obj_dict=prediction_queue)

        model_queue = self._microservice.retrieve(
            "ModelQueue", pk=prediction_queue["model_queue_id"])
        model = self._microservice.retrieve(
            model_class="DescriptionModel", pk=model_queue['model_id'])

        print('Geting model parameters...')
        data = self.get_prediction_data(
            prediction_queue=prediction_queue)
        estimation_cols_info = model_queue['model_matrix_columns'] or {}
        predicton_cols_info = prediction_queue['matrix_columns'] or {}
        output_var_attribute_id = model["output_var_attribute_id"]
        model_parameter = model['parameters']
        prediction_parameters = prediction_queue["parameters"]

        columns_info = predicton_cols_info.get("columns_info", {})
        columns_input = {}
        columns_output = {}
        for name, item in columns_info.items():
            if item["pk"] == output_var_attribute_id and \
                    item["model_class"] == "ModelVarAttribute":
                columns_output[name] = item
            else:
                columns_input[name] = item

        if len(columns_output) == 0:
            if model["type"] == "dummy-model":
                output_var = self._microservice.retrieve(
                    "ModelVarAttribute", pk=output_var_attribute_id)
                columns_output["value"] = output_var

        extra_file = self.load_prediction_extra_file(
            prediction_queue=prediction_queue)
        estimated_model = self.load_model_file(model_queue=model_queue)

        print('Running prediction...')
        try:
            prediction_return = self.run_prediction(
                estimated_model=estimated_model, data=data,
                columns_input=columns_input,
                columns_output=columns_output,
                predicton_cols_info=predicton_cols_info,
                estimation_cols_info=estimation_cols_info,
                output_var_attribute_id=output_var_attribute_id,
                model_parameter=model_parameter,
                prediction_parameter=prediction_parameters,
                extra_file=extra_file, microservice=self._microservice)
        except Exception as e:
            prediction_queue["status"] = "error_prediction"
            prediction_queue["msg"] = str(e)
            self._microservice.save(prediction_queue)
            raise e

        df_results = prediction_return["results"]
        df_results = self.ajust_dataframe_with_description(df_results)
        prediction_queue["status"] = "saving_transformed"
        prediction_queue = self._microservice.save(prediction_queue)

        print('# Saving results transformed data')
        try:
            trans_pred_result = self.save_transformated_prediction(
                transformed_data=df_results.copy(),
                prediction_queue=prediction_queue, model=model)
        except Exception as e:
            prediction_queue["status"] = "error_saving_transformed"
            prediction_queue["msg"] = str(e)
            self._microservice.save(prediction_queue)
            raise e

        prediction_queue["results"].update(trans_pred_result)
        prediction_queue["status"] = "saving_clean"
        prediction_queue = self._microservice.save(prediction_queue)
        print('# Saving results clean data')
        try:
            if len(columns_output):
                clean_pred_result = self.save_clean_prediction(
                    columns_output=columns_output,
                    prediction_return=prediction_return, model=model,
                    prediction_queue=prediction_queue)
        except Exception as e:
            clean_pred_result = {"error_saving_clean": str(e)}

        prediction_queue["results"].update(clean_pred_result)
        prediction_queue["status"] = "waiting_processing"
        prediction_queue["end_prediction"] = datetime.datetime.utcnow()
        prediction_queue = self._microservice.save(prediction_queue)

        shutil.rmtree("temp/")
        return prediction_queue

    def get_estimation_rawdata_summary(self, model_queue: dict,
                                       output_attribute_ids: List[int]):
        """
        Return a summary of the outputs attributes max time.

        Args:
            model_queue (dict): ModelQueue from which to get raw data database.
            output_attribute_ids (list[int]): List of the output attribute ids.
        """
        if model_queue["raw_data_file"] is None:
            return None

        file_path = "temp/rawdata.db"
        with open(file_path, "wb") as file:
            self._storage_object.download_to_file(
                file_path=model_queue["raw_data_file"], file_obj=file)
        rawdata_db_con = create_engine('sqlite:///{}'.format(file_path))
        to_query = ",".join([str(x) for x in output_attribute_ids])
        max_estimation_time = pd.read_sql(
            rawdata_summary_query.format(attributes=to_query),
            con=rawdata_db_con, parse_dates=["max_time"])
        return max_estimation_time

    def get_estimation_data(self, model_queue: dict):
        """
        Get data to estimate model.

        Args:
            model_queue (dict): ModelQueue serialized queue data.
        """
        if model_queue["model_matrix_file"] is None:
            return pd.DataFrame()

        if not os.path.exists("temp/"):
            os.makedirs("temp/")

        file_path = "temp/model_matrix_file.parquet"
        with open(file_path, "wb") as file:
            self._storage_object.download_to_file(
                file_path=model_queue["model_matrix_file"], file_obj=file)

        return_data = pd.read_parquet(file_path)
        os.remove(file_path)

        return return_data

    def get_prediction_data(self, prediction_queue: dict):
        """
        Get data to make predictions.

        Args:
            prediction_queue (dict): PredictionQueue object serialized data.
        """
        if prediction_queue["matrix_file"] is None:
            return pd.DataFrame()

        if not os.path.exists("temp/"):
            os.makedirs("temp/")

        file_path = "temp/matrix_file.gzip"
        with open(file_path, "wb") as file:
            self._storage_object.download_to_file(
                file_path=prediction_queue["matrix_file"], file_obj=file)

        return pd.read_csv(
            file_path, compression="gzip", parse_dates=["time"])

    def load_model_file(self, model_queue: dict):
        """
        Load model file from pumpwood.

        Args:
            model_queue (dict): ModelQueue serialized data.
        """
        model_file = model_queue["model_file"]
        if model_file is None:
            return None

        file_path = "temp/model_file.pickle"
        with open(file_path, "wb") as file:
            self._storage_object.download_to_file(
                file_path=model_queue["model_file"], file_obj=file)
        return pickle.load(open(file_path, "rb"))

    def load_prediction_extra_file(self, prediction_queue: dict):
        """
        Save prediction extra file to local if it exists.

        Args:
            prediction_queue(dict): Serialized prediction queue object.
        Kwargs:
            No kwargs.
        Returns:
            A file IO if there is an extra file associated with prediction
            queue or None with it doesn't.
        """
        extra_file = prediction_queue["extra_file"]
        if extra_file is None:
            return None

        file_name = ntpath.basename(extra_file)
        file_path = self._microservice.retrieve_streaming_file(
            model_class="PredictionQueue", pk=prediction_queue["pk"],
            file_field="extra_file", file_name=file_name, save_path="temp/",
            if_exists="overide")
        return open(file_path, "rb")

    def save_model_file(self, model_queue: dict, model_fit: any):
        """
        Save model file and mark queue as finished.

        Args:
            queue (dict): Serialized object of the model queue.
            model_fit (any): Model fitted.
        """
        end_estimation = datetime.datetime.utcnow()
        model_fit_dump = self.save_temp_file(
            content=pickle.dumps(model_fit),
            suflix='model_file', extention='pickle', binary=True)

        model_queue_obj = self._microservice.retrieve(
            model_class="ModelQueue", pk=model_queue['pk'])
        model_queue_obj['status'] = 'finished_estimation'
        model_queue_obj['end_estimation'] = end_estimation

        if model_fit is not None:
            return self._microservice.save(
                obj_dict=model_queue_obj, files={
                    'model_file': model_fit_dump})
        else:
            return self._microservice.save(
                obj_dict=model_queue_obj)

    def save_residuals(self, model_queue: dict, residuals: pd.DataFrame):
        """
        Save model residual and fitted data.

        Args:
            queue (dict): Serialized object of the model queue.
            residuals (pd.DataFrame): A dataframe with columns:
                time (pd.Datatime): Time index for the residuals or fit.
                modeling_unit_id (int): Pk of the modeling unit index.
                geoarea_id (int): Pk of the geoarea index.
                var_attribute_id (int): Pk of the attribute variable index.
                type (str[20]): Retricted to max 20 characters ex.: (fitted,
                    residual_desviance).
                value: Value of the type of residuals or fit
        """
        residuals["queue_id"] = model_queue["pk"]
        return self._microservice.parallel_bulk_save(
            model_class="ModelQueueResultsResiduals",
            data_to_save=residuals)

    def save_parameters_simple(self, model_queue: dict,
                               parameters_simple: pd.DataFrame):
        """
        Save model simple parameter results.

        Args:
            queue (dict): Serialized object of the model queue.
            parameters_simple (pd.Dataframe): A dataframe with the columns:
                name (str[30]): Name of the parameter.
                type (str[30]): Type of the parameter. Some examples:
                    exogenous_parameter, intercept, converged, model_selection
                time (pd.Timestamp or None): Time index of the parameter.
                modeling_unit_id (int or None): Index of the modeling unit.
                geoarea_id (int or None): Index of the geoarea.
                var_attribute_id (int or None): Index of the attribute.
                var_geoattribute_id (int or None): Index of the geoattribute.
                var_calendar_id (int or None): Index of the calendar.
                var_dummy_id (int or None): Index of the dummy.
                value (dict): Dictionary with results for the parameter.
        Kwargs:
            No kwargs.
        Raise:
            No raises.
        """
        if len(parameters_simple) != 0:
            for queue in parameters_simple:
                queue["queue_id"] = model_queue["pk"]
            return self._microservice.parallel_save(
                list_obj_dict=parameters_simple)

    def save_parameters_complex(self, model_queue: dict,
                                parameters_complex: list):
        """
        Save model simple complex results.

        Args:
            queue (dict): Serialized object of the model queue.
            parameters_complex (list[dict]): A list of dictonary to be used
                in save function of PumpWood microservice. The dicts have keys:
                    obj_dict (dict): with values to be used in saving complex
                        parameters.
                            name (str[50]): name of the complex parameter. Ex.:
                                covar_matrix.
                            data (dict): Dictionary to be saved as a JSON at
                                complex parameter object.
                    file (File Stream or None): A file to be saved in complex
                        parameter object.
        Kwargs:
            No kwargs.
        Raise:
            No raises.
        """
        for complex_par in parameters_complex:
            obj_dict = complex_par['obj_dict']
            obj_dict["queue_id"] = model_queue["pk"]
            if complex_par.get("file") is not None:
                self._microservice.save(
                    obj_dict=obj_dict,
                    files={"file": complex_par["file"]})
            else:
                self._microservice.save(
                    obj_dict=obj_dict)

    def ajust_dataframe_with_description(self, data: pd.DataFrame):
        """
        Remove columns with description.

        Dummy model make those...
        """
        errors = []
        if "modeling_unit" in data.columns:
            unique_modeling_unit = data["modeling_unit"].unique()
            mu_in_datalake = self._microservice.list_without_pag(
                "DescriptionModelingUnit", filter_dict={
                    "description__in": unique_modeling_unit})
            dict_mu = dict(
                [(x["description"], x["pk"]) for x in mu_in_datalake])

            dl_set_mu = set(dict_mu.keys())
            set_modeling_unit = set(unique_modeling_unit)
            missing_mu = set_modeling_unit - dl_set_mu

            # Check if there is any missing modeling units
            if len(missing_mu):
                template = "some modeling_units in prediction are not at " + \
                    "datalake: {}"
                errors.append(template.format(missing_mu))
            else:
                data["modeling_unit_id"] = data[
                    "modeling_unit"].map(dict_mu)
                del data["modeling_unit"]

        if "geoarea" in data.columns:
            unique_geoarea = data["geoarea"].unique()
            geo_in_datalake = self._microservice.list_without_pag(
                "DescriptionGeoarea", filter_dict={
                    "description__in": unique_geoarea})
            dict_geo = dict(
                [(x["description"], x["pk"]) for x in geo_in_datalake])

            dl_set_geo = set(dict_geo.keys())
            set_geoarea = set(unique_geoarea)
            missing_geo = set_geoarea - dl_set_geo

            # Check if there is any missing geoareas
            if len(missing_geo):
                template = "some geoareas in prediction are not at" + \
                    " datalake: {}"
                errors.append(template.format(missing_geo))
            else:
                data["geoarea_id"] = data[
                    "geoarea"].map(dict_geo)
                del data["geoarea"]

        if len(errors):
            raise Exception("\n".join(errors))

        return data

    def save_transformated_prediction(self, transformed_data: pd.DataFrame,
                                      prediction_queue: dict,
                                      model: dict):
        """
        Save transformed prediction.

        Args:
            transformed_data (pd.Dataframe): Data frame with preciton results
                with transformed variable. Columns:
                    time (pd.Timestamp): Time index of the predicton.
                    modeling_unit (int): Index of the modeling_unit.
                    geoarea (int): Index of the geoarea.
                    prediction_type (str[20]): Identification string of the
                        prediction type.
                    value (float): value of the predicton.
            prediction_queue (dict[PredictionQueue]): Serialized object of
                PredictionQueue.
            model (dict[DescriptionModel]) Serialized object of
                DescriptionModel.
        """
        transformed_data["prediction_queue_id"] = prediction_queue["pk"]
        transformed_data["scenario_id"] = prediction_queue[
            "scenario_id"]
        transformed_data["model_id"] = model['pk']
        transformed_data["model_queue_id"] = prediction_queue[
            "model_queue_id"]
        transformed_data["var_attribute_id"] = model[
            "output_var_attribute_id"]

        # Saving data
        self._microservice.parallel_bulk_save(
            model_class="ToLoadPredictionData",
            data_to_save=transformed_data)

        return {
            "toloadpredictiondata_count": len(transformed_data)}

    def inverse_prediction_transf(self, transformated_prediction: pd.DataFrame,
                                  output_variable: dict,
                                  time_frequency: pd.Series):
        """
        Run the inverse transformation of the prediction.

        Args:
            prediction_results (pd.DataFrame): Dataframe with columns.
                time (pd.Timestamp): Time index of the prediction.
                modeling_unit_id (int): Index of the modeling_unit.
                geoarea_id (int): Index of the geoarea.
                prediction_type (str[20]): Identification of the time of the
                    prediction. Ex: ['mean', 'q25', 'q50', 'q75', ...].
                value (float): value of the prediction for the index and the
                    prediction_type.
            output_variable (DescriptionModel): Serialized object of
                description model.
        Kwargs:
            time_frequency (pd.Series[pd.Timestamp]): Serie of times of the
                series.
        Returns:
            pd.DataFrame:
                time (pd.Timestamp): Time index.
                geoarea_id (int): Geoarea index.
                modeling_unit_id (int): Modeling unit index.
                attribute_id (int): Attribute index.
                prediction_type (str): Prediction type index.
                value (float): Value at the previus index.
        """
        inverse_result = self._microservice.execute_action(
            "DataTransformation", action="inverse_transform_data",
            parameters={
                "data": transformated_prediction,
                "column_info": output_variable,
                "time_frequency": time_frequency})["result"]["data"]

        inverse_result = pd.DataFrame(inverse_result)
        inverse_result["attribute_id"] = output_variable["attribute_id"]
        inverse_result["time"] = pd.to_datetime(inverse_result["time"])
        return inverse_result

    def remove_real_data_from_prediction(self, clean_prediction: pd.DataFrame,
                                         prediction_queue: dict):
        """
        Remove real data from prediciton using estimation database.

        Args:
            clean_prediction (pd.DataFrame): Dataframe with untransformated
                data. Dataframe with columns:
                    time (pd.Timestamp): Prediction time index.
                    modeling_unit_id (int): Modeling unit index.
                    geoarea_id (int) Geoarea index.
                    attribute_id (int): Attribute index.
                    prediction_type (str[20]): Prediction type (mean, q25,
                        q75, etc...).
                    value (float): Untransformated prediction.
        return:
            Data with time index that are after that used in estimation
            database or that do not have correspondent. Columns:
                time (pd.Timestamp): Prediction time index.
                modeling_unit_id (int): Modeling unit index.
                geoarea_id (int) Geoarea index.
                attribute_id (int): Attribute index.
                prediction_type (str[20]): Prediction type (mean, q25,
                    q75, etc...).
                value (float): Untransformated prediction.
        """
        model_queue = self._microservice.retrieve(
            "ModelQueue", pk=prediction_queue["model_queue_id"])
        output_attribute_ids = clean_prediction[
            "attribute_id"].unique().tolist()
        max_estimation_time = self.get_estimation_rawdata_summary(
            model_queue=model_queue, output_attribute_ids=output_attribute_ids)

        if max_estimation_time is None:
            return clean_prediction

        merged_max_estimation = clean_prediction.merge(
            max_estimation_time, how="left")
        index_after_estimation = merged_max_estimation["time"] >= \
            merged_max_estimation["max_time"]
        index_not_on_estimation = merged_max_estimation["max_time"].isna()
        prediction = merged_max_estimation[
            index_after_estimation & ~index_not_on_estimation]

        del prediction["max_time"]
        return prediction

    def save_clean_prediction(self, columns_output: dict,
                              prediction_return: pd.DataFrame, model: dict,
                              prediction_queue:  dict):
        """
        Save clean prediction.

        Args:
            columns_output (dict): Dictionary with descriptions of output
                columns.
            prediction_result (dict): A dictionary with keys.
                results Data (pd.Dataframe): frame with preciton results.
                    time (pd.Timestamp): Time index of the predicton.
                    modeling_unit (int): Index of the modeling_unit.
                    geoarea (int): Index of the geoarea.
                    prediction_type (str[20]): Identification string of the
                        prediction type.
                    value (float): value of the predicton.
                notinv_prediction_type (list[str]): List of the prediction_type
                    that won't be inverted and saved at clean prediction.
            model (dict[DescriptionModel]): Serialized object of
                DescriptionModel.
            prediction_queue (dict[PredictionQueue]): Serialized object of
                PredictionQueue.
        Return:
            dict: With toloadcleanpredictiondata_count key with number of
                prediction data that were produced.
        """
        transformed_data = prediction_return["results"]
        notinv_prediction_type = prediction_return["notinv_prediction_type"]
        if len(transformed_data) != 0:
            if len(columns_output) != 1:
                raise Exception(
                    "Save clean prediction not implemented for more than one" +
                    " output column.")
            index_to_invert = ~transformed_data["prediction_type"].isin(
                notinv_prediction_type)
            data_to_invert = transformed_data[index_to_invert]
            out_key = list(columns_output.keys())[0]

            clean_prediction = self.inverse_prediction_transf(
                transformated_prediction=data_to_invert.copy(),
                output_variable=columns_output[out_key],
                time_frequency=None)
            clean_prediction["attribute_id"] = columns_output[
                out_key]["attribute_id"]
            prediction_data = self.remove_real_data_from_prediction(
                clean_prediction=clean_prediction,
                prediction_queue=prediction_queue)

            prediction_data["prediction_queue_id"] = prediction_queue["pk"]
            prediction_data["scenario_id"] = prediction_queue["scenario_id"]
            prediction_data["model_id"] = prediction_queue["model_id"]
            prediction_data["model_queue_id"] = prediction_queue[
                "model_queue_id"]
            ############################################
            # Saving data
            self._microservice.parallel_bulk_save(
                model_class="ToLoadCleanPredictionData",
                data_to_save=prediction_data)
            ############################################
            return {
                "toloadcleanpredictiondata_count": len(prediction_data)}
        else:
            return {
                "toloadcleanpredictiondata_count": 0}

    ############################
    # Funcitons to be overiden #
    @classmethod
    @abstractmethod
    def run_model(cls, data: pd.DataFrame, columns_info: dict,
                  output_var_attribute_id: int, model_parameter: dict):
        """
        Run model and returns the results.

        Results are for residuals, simple parameters and complex parameters
        (files and JSON).

        Args:
            data (pd.DataFrame): Model matrix to be used on model.
            columns_info (dict): Info of the columns in Model matrix.
            output_var_attribute_id (int): pk of the variable attribute to be
                used as exogenous variable.
            model_parameter (dict): parameters to used in model running.
        Returns:
            model (any): The object of the fitted model.
            residuals (pd.DataFrame): A dataframe with columns:
                time (pd.Datatime): Time index for the residuals or fit.
                modeling_unit_id (int): Pk of the modeling unit index.
                geoarea_id (int): Pk of the geoarea index.
                var_attribute_id (int): Pk of the attribute variable index.
                type (str[20]): Retricted to max 20 characters ex.: (fitted,
                    residual_desviance).
                value: Value of the type of residuals or fit
            parameters_simple (pd.Dataframe): A dataframe with the columns:
                name (str[30]): Name of the parameter.
                type (str[30]): Type of the parameter. Some examples:
                    exogenous_parameter, intercept, converged, model_selection
                time (pd.Timestamp or None): Time index of the parameter.
                modeling_unit_id (int or None): Index of the modeling unit.
                geoarea_id (int or None): Index of the geoarea.
                var_attribute_id (int or None): Index of the attribute.
                var_geoattribute_id (int or None): Index of the geoattribute.
                var_calendar_id (int or None): Index of the calendar.
                var_dummy_id (int or None): Index of the dummy.
                value (dict): Dictionary with results for the parameter.
            parameters_complex (list[dict]): A list of dictonary to be used
                in save function of PumpWood microservice. The dicts have keys:
                    obj_dict (dict): with values to be used in saving complex
                        parameters.
                            name (str[50]): name of the complex parameter. Ex.:
                                covar_matrix.
                            data (dict): Dictionary to be saved as a JSON at
                                complex parameter object.
                    file (File Stream or None): A file to be saved in complex
                        parameter object.
        Kwargs:
            No Kwargs.

        Raises:
            No particular raises.
        """
    pass

    @classmethod
    @abstractmethod
    def get_model_parameters(cls, partial_parameters: dict = {}):
        """
        Return the parameters necessary to run the model.

        This method must be implemented for each model, it receives the partial
        parameters of the model (before saving object). And returns possible
        combination of parameters that make sense to build the model (using
        the actual partial_parameters).

        Args:
            partial_parameters (dict): key-value dictonary with the parameters
                to be used in model building.
        Returns:
            required (dict): A dictionary with parameters keys with the
                description of the values that can used in model. All
                parameters returned on this dict are considered necessary.
            optional (dict): A dictionary with parameters keys with the
                description of the values that can used in model. All
                parameters returned on this dict are considered NOT necessary.
        Raise:
            No particular errors.
        """
        pass

    @classmethod
    @abstractmethod
    def run_prediction(self, estimated_model: any, data: pd.DataFrame,
                       predicton_cols_info: dict, estimation_cols_info: dict,
                       output_var_attribute_id: int,
                       model_parameter: dict, prediction_parameter: dict,
                       extra_file: io.TextIOWrapper,
                       microservice: PumpWoodMicroService):
        """
        Create model prediction using data matrix build.

        Args:
            estimated_model (any): An estimated statsmodels object.
            data (pd.DataFrame): Model matrix to be used on model.
            predicton_cols_info (dict): With two keys:
                columns_order (list): Order of the columns in matrix.
                columns_info (dict): A dictionary with info of the columns.
            estimation_cols_info (dict): With two keys:
                columns_order (list): Order of the columns in matrix.
                columns_info (dict): A dictionary with info of the columns.
            output_var_attribute_id (int): pk of the variable attribute to be
                used as exogenous variable.
            model_parameter (dict): parameter used in model running.
            prediction_parameter (dict): parameters for prediction.
            extra_file (io.TextIOWrapper): Io for the predicton extra file.
            microservice (PumpWoodMicroService): Microservice object to be
                used on run prediction.
        Returns:
            notinv_prediction_type (list[str]): list of prediction_type to be
                inverted when making prediction.
            results (pd.Dataframe): A dataframe with columns.
                time (pd.Timestamp): Time index of the prediction.
                modeling_unit_id (int): Index of the modeling_unit.
                geoarea_id (int): Index of the geoarea.
                prediction_type (str[20]): Identification of the time of the
                    prediction. Ex: ['mean', 'q25', 'q50', 'q75', ...].
                value (float): value of the prediction for the index and the
                    prediction_type.
        Kwargs:
            No Kwargs.

        Raises:
            No particular raises.
        """
        pass
