# Importando bibliotecas
import os
from datetime import datetime 
from feast import (Entity,
                   FeatureService,
                   FeatureView,
                   Field,
                   FileSource)

from feast.types import String, Int64

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
adult_hist_social_fv = FeatureView(name = "adult_hist_social_data",
                              entities = [adult_entity],
                              schema = [Field(name = "age", dtype = Int64),
                              Field(name = "race", dtype = String),
                              Field(name = "education", dtype = String),
                              Field(name = "education-num", dtype = Int64),
                              Field(name = "marital-status", dtype = String),
                              Field(name = "sex", dtype = String),
                              Field(name = "native-country", dtype = String)],
                              source = adult_hist_db_file_source)

adult_hist_income_fv = FeatureView(name = "adult_hist_income_data",
                              entities = [adult_entity],
                              schema = [Field(name = "workclass", dtype = String),
                              Field(name = "occupation", dtype = String),
                              Field(name = "capital-gain", dtype = Int64),
                              Field(name = "capital-loss", dtype = Int64),
                              Field(name = "income", dtype = String)],
                              source = adult_hist_db_file_source)

# Definindo File Source - Dados para Inferência
adult_inf_db_file_source = FileSource(name = "adult_inf_db_source",
                                      path = os.path.join(path_silver, "adult_dataset_inf.parquet"),
                                      timestamp_field = "event_timestamp",
                                      )

# Definindo Feature Views - Dados para Inferência
adult_inf_social_fv = FeatureView(name = "adult_inf_social_data",
                              entities = [adult_entity],
                              schema = [Field(name = "age", dtype = Int64),
                              Field(name = "race", dtype = String),
                              Field(name = "education", dtype = String),
                              Field(name = "education-num", dtype = Int64),
                              Field(name = "marital-status", dtype = String),
                              Field(name = "sex", dtype = String),
                              Field(name = "native-country", dtype = String)],
                              source = adult_inf_db_file_source)

adult_inf_income_fv = FeatureView(name = "adult_inf_income_data",
                              entities = [adult_entity],
                              schema = [Field(name = "workclass", dtype = String),
                              Field(name = "occupation", dtype = String),
                              Field(name = "capital-gain", dtype = Int64),
                              Field(name = "capital-loss", dtype = Int64)],
                              source = adult_inf_db_file_source)

# Definindo Feature Services - Treino Modelo v1
adult_income_fs_train_v1 = FeatureService(name = "adult_income_train_v1",
                                          features = [adult_hist_social_fv[["race", "education"]],
                                                     adult_hist_income_fv])

# Definindo Feature Services - Inferência Modelo v1
adult_income_fs_inf_v1 = FeatureService(name = "adult_income_inf_v1",
                                        features = [adult_inf_social_fv[["race", "education"]],
                                                    adult_inf_income_fv])

# Definindo Feature Services - Treino Modelo v2
adult_income_fs_train_v2 = FeatureService(name = "adult_income_train_v2",
                                          features = [adult_hist_social_fv,
                                                     adult_hist_income_fv])

# Definindo Feature Services - Inferência Modelo v2
adult_income_fs_inf_v2 = FeatureService(name = "adult_income_inf_v2",
                                        features = [adult_inf_social_fv,
                                                    adult_inf_income_fv])