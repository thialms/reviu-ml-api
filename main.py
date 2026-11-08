from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Inicializa a API
app = FastAPI(title="Reviu ML API", description="Microserviço de predição de revisões")

# Carrega o modelo na memória apenas uma vez quando a API inicia
try:
    modelo = joblib.load("reviu_ml_model_v2.joblib")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    modelo = None

# Define o "contrato" da API (exatamente o que o Spring Boot vai mandar)
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

@app.post("/prever_intervalo")
def prever_intervalo(dados: RevisaoRequest):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo de Machine Learning não carregado.")

    # 1. Transforma o JSON recebido em um DataFrame do Pandas (formato que o Scikit-Learn entende)
    df = pd.DataFrame([dados.dict()])
    
    # 2. Faz a predição
    previsao = modelo.predict(df)[0]
    
    # 3. Retorna o resultado arredondado
    return {"proximo_intervalo_dias": round(previsao)}