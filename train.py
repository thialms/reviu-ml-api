import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib

# ==========================================
# 1. GERAÇÃO DE DADOS MOCKADOS (CENÁRIO FICTÍCIO)
# ==========================================
np.random.seed(42)
num_registros = 5000

# Simulando as respostas do onboarding do Reviu (Cold Start) baseadas nas telas
focos = ['Vestibular / ENEM', 'Concursos Públicos', 'Idiomas', 'Faculdade', 'Aprendizado geral']
prazos = ['Próximos 3 meses', 'Entre 3 e 12 meses', 'Sem prazo fixo / Longo prazo']
metas_diarias = ['10 minutos', '30 minutos', '1 hora', 'Mais de 1 hora']
horarios_foco = ['Manhã', 'Tarde', 'Noite', 'Flexível']
retencao_conteudo = ['Esqueco fácil', 'Equilibrado', 'Retenho bem']
ritmos_revisao = ['Leve', 'Equilibrado', 'Intenso']

dados = {
    'id_usuario': np.random.randint(1, 100, num_registros),
    'foco_estudos': np.random.choice(focos, num_registros),
    'prazo_final': np.random.choice(prazos, num_registros),
    'meta_diaria': np.random.choice(metas_diarias, num_registros),
    'horario_foco': np.random.choice(horarios_foco, num_registros),
    'retencao_conteudo': np.random.choice(retencao_conteudo, num_registros),
    'ritmo_revisao': np.random.choice(ritmos_revisao, num_registros),
    
    # Dados da performance no flashcard (Dinâmicos após o uso)
    'repeticoes_anteriores': np.random.randint(0, 10, num_registros),
    'facilidade_atual': np.random.uniform(1.3, 2.5, num_registros), # Fator SM-2
    'nota_revisao': np.random.randint(1, 5, num_registros), # 1(Errou) a 4(Fácil)
}

df = pd.DataFrame(dados)

# Criando a variável "Alvo" (Target): O próximo intervalo ideal em dias.
# Lógica simulada para gerar os dados:
df['proximo_intervalo_dias'] = (
    df['repeticoes_anteriores'] * df['facilidade_atual'] * (df['nota_revisao'] / 2)
)

# Ajustes lógicos baseados no perfil do utilizador:
# Quem esquece fácil tem intervalos menores
df.loc[df['retencao_conteudo'] == 'Esqueco fácil', 'proximo_intervalo_dias'] *= 0.7 
# Ritmo intenso diminui os intervalos
df.loc[df['ritmo_revisao'] == 'Intenso', 'proximo_intervalo_dias'] *= 0.8
# Prazos curtos exigem revisões mais frequentes
df.loc[df['prazo_final'] == 'Próximos 3 meses', 'proximo_intervalo_dias'] *= 0.75

df['proximo_intervalo_dias'] = np.ceil(df['proximo_intervalo_dias']).clip(lower=1)


# ==========================================
# 2. PREPARAÇÃO DOS DADOS E TREINAMENTO
# ==========================================
X = df.drop(columns=['id_usuario', 'proximo_intervalo_dias'])
y = df['proximo_intervalo_dias']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tratando as variáveis categóricas (as respostas do app)
features_categoricas = ['foco_estudos', 'prazo_final', 'meta_diaria', 'horario_foco', 'retencao_conteudo', 'ritmo_revisao']
features_numericas = ['repeticoes_anteriores', 'facilidade_atual', 'nota_revisao']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', features_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('modelo', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("A treinar o modelo preditivo do Reviu...")
pipeline.fit(X_train, y_train)

# ==========================================
# 3. AVALIAÇÃO E EXPORTAÇÃO
# ==========================================
previsoes = pipeline.predict(X_test)
erro = mean_absolute_error(y_test, previsoes)
print(f"Erro Médio Absoluto: {erro:.2f} dias")

# Testando um cenário real com base nas opções do app:
aluno_novo = pd.DataFrame({
    'foco_estudos': ['Concursos Públicos'],
    'prazo_final': ['Próximos 3 meses'],
    'meta_diaria': ['Mais de 1 hora'],
    'horario_foco': ['Noite'],
    'retencao_conteudo': ['Esqueco fácil'],
    'ritmo_revisao': ['Intenso'],
    'repeticoes_anteriores': [1],
    'facilidade_atual': [2.0],
    'nota_revisao': [3] # Achou bom, mas não tão fácil
})

previsao_novo = pipeline.predict(aluno_novo)
print(f"\nPrevisão para o aluno teste: O flashcard deve ser revisto em {previsao_novo[0]:.0f} dias.")

joblib.dump(pipeline, 'reviu_ml_model_v2.joblib')
print("\nModelo guardado com sucesso como 'reviu_ml_model_v2.joblib'!")