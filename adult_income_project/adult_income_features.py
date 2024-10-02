# Importando bibliotecas
import os
from datetime import datetime 
from feast import (Entity,
                   FeatureService,
                   FeatureView,
                   Field,
                   FileSource,
                   PushSource,
                   RequestSource)

from feast.types import String, Float64, Float32, Int64

# Definindo paths
path_codigos = os.getcwd()
path_projeto = os.path.dirname(path_codigos)
path_data = os.path.join(path_projeto, "02_data")
path_bronze = os.path.join(path_data, "01_bronze")
path_silver = os.path.join(path_data, "02_silver")

#### Setup Feature Store

# Criando entity
adult_entity = Entity(name = "adult", join_keys = ["adult_id"])

# Definindo File Source - Dados Históricos
adult_hist_db_file_source = FileSource(name = "adult_hist_db_source",
                                      path = os.path.join(path_silver, "adult_dataset_hist.parquet"),
                                      timestamp_field = "event_timestamp",
                                      )

# Definindo Feature Views - Dados Históricos
adult_hist_social_dv = FeatureView(name = "adult_hist_social_data",
                              entities = [adult_entity],
                              schema = [Field(name = "age", dtype = INT64),
                              Field(name = "race", dtype = STRING),
                              Field(name = "education", dtype = STRING),
                              Field(name = "education-num", dtype = INT64),
                              Field(name = "marital-status", dtype = STRING),
                              Field(name = "sex", dtype = STRING),
                              Field(name = "native-country", dtype = STRING)],
                              source = adult_hist_db_file_source)

adult_hist_income_dv = FeatureView(name = "adult_hist_income_data",
                              entities = [adult_entity],
                              schema = [Field(name = "workclass", dtype = string),
                              Field(name = "occupation", dtype = STRING),
                              Field(name = "capital-gain", dtype = INT64),
                              Field(name = "capital-loss", dtype = INT64),
                              Field(name = "income", dtype = STRING)],
                              source = adult_hist_db_file_source)

# Definindo File Source - Dados para Inferência
adult_inf_db_file_source = FileSource(name = "adult_inf_db_source",
                                      path = os.path.join(path_silver, "adult_dataset_inf.parquet"),
                                      timestamp_field = "event_timestamp",
                                      )

# Definindo Feature Views - Dados para Inferência
adult_inf_social_dv = FeatureView(name = "adult_inf_social_data",
                              entities = [adult_entity],
                              schema = [Field(name = "age", dtype = INT64),
                              Field(name = "race", dtype = STRING),
                              Field(name = "education", dtype = STRING),
                              Field(name = "education-num", dtype = INT64),
                              Field(name = "marital-status", dtype = STRING),
                              Field(name = "sex", dtype = STRING),
                              Field(name = "native-country", dtype = STRING)],
                              source = adult_inf_db_file_source)

adult_inf_income_dv = FeatureView(name = "adult_inf_income_data",
                              entities = [adult_entity],
                              schema = [Field(name = "workclass", dtype = string),
                              Field(name = "occupation", dtype = STRING),
                              Field(name = "capital-gain", dtype = INT64),
                              Field(name = "capital-loss", dtype = INT64)],
                              source = adult_inf_db_file_source)

# Definindo Feature Services - Treino Modelo v1
adult_income_fs_train_v1 = FeatureService(name = "adult_income_train_v1",
                                          features = [adult_hist_social_dv[["race", "education"]],
                                                     adult_hist_income_dv])

# Definindo Feature Services - Inferência Modelo v1
adult_income_fs_inf_v1 = FeatureService(name = "adult_income_inf_v1",
                                        features = [adult_inf_social_dv[["race", "education"]],
                                                    adult_inf_income_dv])

# Definindo Feature Services - Treino Modelo v2
adult_income_fs_train_v2 = FeatureService(name = "adult_income_train_v2",
                                          features = [adult_hist_social_dv,
                                                     adult_hist_income_dv])

# Definindo Feature Services - Inferência Modelo v2
adult_income_fs_inf_v2 = FeatureService(name = "adult_income_inf_v2",
                                        features = [adult_inf_social_dv,
                                                    adult_inf_income_dv])