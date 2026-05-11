# Preprocessamento - ICU Length of Stay Prediction

## Objetivo da Parte de Preprocessamento

O objetivo do preprocessamento foi transformar os dados brutos de UCI, originalmente muito grandes e em formato de eventos clínicos, num dataset estruturado e pronto para as fases seguintes do projeto: preparação, treino e avaliação dos modelos.

O problema que queremos resolver é prever a duração da estadia na UCI, ou seja, o `Length of Stay (LOS)`. O target do projeto é a coluna `LOS` da tabela `ICUSTAYS`, que representa a duração total da estadia em dias.

Como queremos simular uma previsão feita no início da estadia, usamos apenas informação disponível nas primeiras 24 horas após a entrada na UCI.

## Dados Utilizados

Nesta fase usamos as seguintes tabelas:

- `ICUSTAYS`: contém a unidade de análise, o `ICUSTAY_ID`, os tempos de entrada e saída da UCI, e o target `LOS`.
- `CHARTEVENTS`: contém medições clínicas feitas ao longo do tempo, como sinais vitais e observações registadas na UCI.
- `D_ITEMS`: contém a descrição dos `ITEMID`s usados em `CHARTEVENTS`.
- `PATIENTS`: contém informação demográfica, como sexo e data de nascimento.
- `ADMISSIONS`: contém contexto da admissão hospitalar, como tipo de admissão, seguro, etnia e estado civil.

## Porque Usamos BigQuery

A tabela `CHARTEVENTS` é muito grande. No nosso caso, depois de carregada para BigQuery, tinha mais de 330 milhões de linhas.

Processar esta tabela localmente com pandas seria lento e pesado em memória. Por isso, usamos o Google Cloud Platform, em particular o BigQuery, como motor de processamento distribuído.

O BigQuery foi usado para:

- carregar os CSVs guardados no Google Cloud Storage;
- filtrar os eventos clínicos pelas primeiras 24 horas;
- remover eventos sem valor numérico;
- selecionar os 50 `ITEMID`s mais frequentes;
- agregar medições ao nível de cada estadia na UCI;
- juntar informação de pacientes e admissões;
- criar uma tabela final compacta, com uma linha por `ICUSTAY_ID`.

Isto responde diretamente ao requisito do professor de usar recursos como BigQuery, pipelines e processamento escalável para analisar dados grandes.

## Pipeline Implementado

O pipeline criado tem duas versões complementares:

- pipeline local em Python: `src/preprocessing/preprocess.py`;
- pipeline escalável em BigQuery: `src/preprocessing/bigquery_preprocess.py` + `sql/bigquery_preprocessing.sql`.

O pipeline BigQuery é o principal para os dados grandes.

O ficheiro de configuração usado é:

```text
configs/bigquery.yaml
```

Nele definimos:

- `project_id`: projeto GCP;
- `dataset_id`: dataset BigQuery;
- bucket onde estão os CSVs;
- prefixo dos ficheiros raw;
- nomes das tabelas;
- janela temporal;
- número de `ITEMID`s selecionados;
- localização do export final.

## Janela Temporal de 24 Horas

Para evitar usar informação do futuro, só usamos eventos clínicos que aconteceram entre:

```text
INTIME
```

e

```text
INTIME + 24 horas
```

Isto significa que o modelo recebe apenas informação inicial da estadia na UCI e tenta prever a duração total da estadia.

Esta decisão é importante para evitar leakage temporal.

## Target

O target é:

```text
LOS
```

Esta coluna vem da tabela `ICUSTAYS` e representa a duração total da estadia na UCI em dias.

Se `LOS` não existisse, poderia ser calculado como:

```text
OUTTIME - INTIME
```

Mas `OUTTIME` não é usado como feature, porque revelaria diretamente informação sobre o futuro.

## Prevenção de Data Leakage

Tivemos cuidado para não usar variáveis que só seriam conhecidas depois da estadia ou no momento da alta.

Não usamos como features:

- `OUTTIME`;
- `DISCHTIME`;
- `DEATHTIME`;
- `DISCHARGE_LOCATION`;
- `HOSPITAL_EXPIRE_FLAG`;
- campos diretamente relacionados com morte ou alta.

Esses campos poderiam dar pistas artificiais ao modelo sobre o tempo de estadia e aumentar falsamente a performance.

## Feature Engineering em CHARTEVENTS

`CHARTEVENTS` está em formato temporal, com várias linhas por paciente, por variável e por momento.

Como os modelos tabulares esperam uma linha por exemplo, agregámos os eventos para obter uma linha por `ICUSTAY_ID`.

Para cada `ITEMID` selecionado, calculámos:

- `mean`: valor médio nas primeiras 24h;
- `min`: valor mínimo nas primeiras 24h;
- `max`: valor máximo nas primeiras 24h;
- `std`: variação/desvio padrão;
- `count`: número de medições;
- `last`: último valor observado dentro da janela.

Exemplo:

```text
item_220045_mean
item_220045_min
item_220045_max
item_220045_std
item_220045_count
item_220045_last
```

Estas features resumem a evolução inicial de sinais clínicos importantes.

## Porque o Dataset Final é Muito Menor

Os CSVs originais são enormes porque têm uma linha por evento clínico. Por exemplo, `CHARTEVENTS` tem centenas de milhões de linhas.

Depois do preprocessamento, passamos a ter:

```text
1 linha por ICUSTAY_ID
```

Além disso:

- usamos só as primeiras 24h;
- usamos só valores numéricos;
- usamos só os top 50 `ITEMID`s;
- agregamos várias medições numa única linha;
- removemos colunas textuais e campos que não entram no modelo.

Por isso, é normal o ficheiro final ter dezenas de MB, apesar dos ficheiros originais terem vários GB.

## Features de Contexto

Além dos eventos clínicos, adicionámos contexto inicial do paciente e da admissão:

- `AGE`;
- `GENDER`;
- `ADMISSION_TYPE`;
- `ADMISSION_LOCATION`;
- `INSURANCE`;
- `LANGUAGE`;
- `RELIGION`;
- `MARITAL_STATUS`;
- `ETHNICITY`;
- `FIRST_CAREUNIT`;
- `LAST_CAREUNIT`;
- `DBSOURCE`;
- `ED_LOS_HOURS`;
- `HAS_CHARTEVENTS_DATA`.

Estas variáveis ajudam o modelo a considerar fatores demográficos e administrativos sem depender de informação futura.

## Outputs Gerados

No BigQuery, são geradas as tabelas:

```text
icu_los.selected_items
icu_los.features_24h
icu_los.quality_report_24h
```

`selected_items` contém os `ITEMID`s selecionados e respetivas descrições.

`features_24h` é a tabela final preprocessada, com uma linha por estadia na UCI.

`quality_report_24h` contém métricas de controlo, como número de eventos válidos, eventos dentro da janela temporal e número de linhas finais.

O ficheiro final exportado para Cloud Storage fica em:

```text
gs://icu-los-prediction-20260510-raw/processed/features_24h_*.csv
```

E depois é descarregado localmente para:

```text
data/processed/bigquery_features
```

## Comandos Usados

Autenticação:

```powershell
& "C:\Users\pavfe\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" auth application-default login
& "C:\Users\pavfe\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" config set project icu-los-prediction-20260510
```

Carregar os CSVs para BigQuery e correr o pipeline:

```powershell
python -m src.preprocessing.bigquery_preprocess --config configs/bigquery.yaml --load-raw --export --download data/processed/bigquery_features
```

Depois de as tabelas já estarem carregadas, não é preciso usar `--load-raw` outra vez:

```powershell
python -m src.preprocessing.bigquery_preprocess --config configs/bigquery.yaml --export --download data/processed/bigquery_features
```

## Resultado

O resultado do preprocessamento é uma tabela compacta, estruturada e adequada para machine learning:

- uma linha por estadia na UCI;
- target `LOS`;
- features clínicas das primeiras 24h;
- features demográficas e de admissão;
- sem informação futura usada como input;
- processada de forma escalável com BigQuery.

## Frase Para a Apresentação

Uma forma simples de explicar esta parte:

> Como a tabela `CHARTEVENTS` é muito grande, com mais de 330 milhões de linhas, usámos BigQuery como motor de processamento distribuído. O BigQuery fez os joins, aplicou a janela temporal das primeiras 24 horas, agregou as medições clínicas e criou uma tabela final com uma linha por estadia na UCI. Assim evitámos processar localmente vários GB de dados e garantimos que o modelo só usa informação disponível no início da estadia.

Outra frase útil:

> O target é o `LOS`, ou seja, a duração total da estadia na UCI. As features foram construídas apenas com dados das primeiras 24 horas, para evitar data leakage e simular uma previsão real feita no início da admissão.
