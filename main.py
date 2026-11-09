import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

app = FastAPI(
    title="Reviu ML API",
    description="Microserviço de predição de revisões adaptativas"
)

# Variável global para guardar o modelo na memória
modelo = None

def carregar_modelo_azure():
    try:
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "rg-reviu-ml-api")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME", "reviu-ml-workspace")
        model_name = os.getenv("AZURE_MODEL_NAME", "reviu-random-forest")
        model_version = os.getenv("AZURE_MODEL_VERSION", "1")

        print(f"Tentando conectar ao Azure ML: {workspace_name}...")
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        print(f"Baixando modelo '{model_name}' v{model_version}...")
        ml_client.models.download(name=model_name, version=model_version, download_path=".")

        caminho_modelo = None
        for root, _, files in os.walk(model_name):
            for file in files:
                if file.endswith(".joblib"):
                    caminho_modelo = os.path.join(root, file)
                    break

        if caminho_modelo and os.path.exists(caminho_modelo):
            print(f"Modelo carregado com sucesso de: {caminho_modelo}")
            return joblib.load(caminho_modelo)
        return None

    except Exception as e:
        print(f"Erro ao carregar modelo do Azure ML: {e}")
        return None

# Carrega o modelo de forma assíncrona/protegida ao subir o servidor
@app.on_event("startup")
def startup_event():
    global modelo
    modelo = carregar_modelo_azure()

class RevisaoRequest(BaseModel):
    foco_estudos: str
    prazo_final: str
    meta_diaria: str
    horario_foco: str
    retencao_conteudo: str
    ritmo_revisao: str
    repeticoes_anteriores: int
    facilidade_atual: float
    nota_revisao: int

@app.get("/")
def health_check():
    return {
        "status": "online",
        "modelo_carregado": modelo is not None
    }

@app.post("/prever_intervalo")
def prever_intervalo(dados: RevisaoRequest):
    if modelo is None:
        raise HTTPException(
            status_code=500, 
            detail="Modelo não carregado. Verifique as credenciais ou logs do Azure ML."
        )
    df = pd.DataFrame([dados.dict()])
    previsao = modelo.predict(df)[0]
    return {"proximo_intervalo_dias": int(round(previsao))}
