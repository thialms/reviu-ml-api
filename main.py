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

# Carrega variáveis de ambiente (com valores padrão para o seu ambiente)
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "COLOQUE_SEU_SUBSCRIPTION_ID_AQUI")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-reviu-ml-api")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "reviu-ml-workspace")
MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "reviu-random-forest")
MODEL_VERSION = os.getenv("AZURE_MODEL_VERSION", "1")

def carregar_modelo_azure():
    """Baixa o modelo do Model Registry no Azure ML Workspace"""
    try:
        print(f"Conectando ao Azure ML Workspace: {WORKSPACE_NAME}...")
        # Usa DefaultAzureCredential: funciona via Azure CLI no PC e via Managed Identity no Web App
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )

        print(f"Baixando modelo '{MODEL_NAME}' versão '{MODEL_VERSION}'...")
        # Baixa o artefato para a pasta local temporária
        ml_client.models.download(name=MODEL_NAME, version=MODEL_VERSION, download_path=".")

        # Procura o arquivo .joblib baixado na estrutura de pastas gerada pelo Azure ML
        caminho_modelo = None
        for root, _, files in os.walk(MODEL_NAME):
            for file in files:
                if file.endswith(".joblib"):
                    caminho_modelo = os.path.join(root, file)
                    break

        if caminho_modelo and os.path.exists(caminho_modelo):
            print(f"Modelo carregado com sucesso de: {caminho_modelo}")
            return joblib.load(caminho_modelo)
        else:
            raise FileNotFoundError("Arquivo .joblib não encontrado após o download.")

    except Exception as e:
        print(f"Erro ao carregar modelo do Azure ML: {e}")
        # Fallback local para desenvolvimento se houver um arquivo local
        if os.path.exists("reviu_ml_model_v2.joblib"):
            print("Carregando cópia local de fallback...")
            return joblib.load("reviu_ml_model_v2.joblib")
        return None

# Carrega o modelo na memória ao iniciar a API
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
        "modelo_carregado": modelo is not None,
        "workspace": WORKSPACE_NAME
    }

@app.post("/prever_intervalo")
def prever_intervalo(dados: RevisaoRequest):
    if modelo is None:
        raise HTTPException(
            status_code=500, 
            detail="Modelo de Machine Learning não está carregado. Verifique as credenciais da Azure."
        )
    
    df = pd.DataFrame([dados.dict()])
    previsao = modelo.predict(df)[0]
    return {"proximo_intervalo_dias": int(round(previsao))}