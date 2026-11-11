import sys
sys.path = [p for p in sys.path if "/agents/python" not in p]
for mod in list(sys.modules.keys()):
    if "typing_extensions" in mod or "anyio" in mod:
        del sys.modules[mod]

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

app = FastAPI(title="Reviu ML API")
modelo = None
erro_detalhado = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def validar_chave_api(api_key: str = Security(api_key_header)):
    chave_esperada = os.getenv("API_SECRET_KEY")
    if not chave_esperada:
        raise HTTPException(status_code=500, detail="Chave de segurança não configurada no Azure.")
    if api_key != chave_esperada:
        raise HTTPException(status_code=403, detail="Acesso negado. Chave inválida.")
    return api_key

def obter_modelo():
    global modelo, erro_detalhado
    if modelo is not None:
        return modelo, None

    try:
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "rg-reviu-ml-api")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME", "reviu-ml-workspace")
        model_name = os.getenv("AZURE_MODEL_NAME", "reviu-random-forest")
        model_version = os.getenv("AZURE_MODEL_VERSION", "1")

        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        ml_client.models.download(name=model_name, version=model_version, download_path=".")

        caminho_modelo = None
        for root, _, files in os.walk(model_name):
            for file in files:
                if file.endswith(".joblib"):
                    caminho_modelo = os.path.join(root, file)
                    break

        if caminho_modelo and os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo, None
        
        erro_detalhado = f"Arquivo .joblib nao encontrado na pasta '{model_name}'"
        return None, erro_detalhado

    except Exception as e:
        erro_detalhado = str(e)
        return None, erro_detalhado

@app.get("/")
def health_check():
    return {
        "status": "online",
        "modelo_carregado": modelo is not None,
    }

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

@app.post("/prever_intervalo", dependencies=[Depends(validar_chave_api)])
def prever_intervalo(dados: RevisaoRequest):
    mod, err = obter_modelo()
    if mod is None:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {err}")
    df = pd.DataFrame([dados.dict()])
    previsao = mod.predict(df)[0]
    return {"proximo_intervalo_dias": int(round(previsao))}
