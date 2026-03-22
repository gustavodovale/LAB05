# Tarefa 1: Preparando o Dataset

from datasets import load_dataset

def carregar_dados_miniatura():
    print("Baixando o dataset opus_books (Inglês-Português)...")
    
    # 1. Carregamos o dataset especificando o idioma do Inglês para o Portugues
    # O parametro split="train": server para pegar apenas a pasta de treinamento oficial
    dataset_completo = load_dataset("opus_books", "en-pt", split="train")
    
    print(f"Tamanho original do dataset: {len(dataset_completo)} frases.")
    
    # 2. Selecionar apenas 1000 registros do dataset completo
    dataset_filtrado = dataset_completo.select(range(1000))
    
    print(f"Tamanho filtrado selecionada: {len(dataset_filtrado)} frases.\n")

    exemplo = dataset_filtrado[0]['translation']

    exemplo_ingles = exemplo['en']

    exemplo_portugues = exemplo['pt']

    print("Exemplo de dado carregado:")
    print(f"🇺🇸 Inglês: {exemplo_ingles}")
    print(f"🇧🇷 Português: {exemplo_portugues}")
    
    return dataset_filtrado


# Teste da tarefa 1
#if __name__ == "__main__":
    #meus_dados = carregar_dados_miniatura()
    
    # teste de funcionamento dos dados carregados
    # exemplo = meus_dados[0]['translation']
    # print("Exemplo de dado carregado:")
    # print(f"🇺🇸 Inglês: {exemplo['en']}")
    # print(f"🇧🇷 Português: {exemplo['pt']}")


# Tarefa 2: Tokenização Básica

import torch

from transformers import AutoTokenizer

def tokenizar_dataset(dataset_filtrado):
    print("Carregando o Tokenizador do BERT Multilíngue...")
    
    # 1. Importando o tokenizador pronto do Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Definimos um tamanho fixo para as matrizes (Padding)
    MAX_LEN = 32 
    
    # Listas para armazenar as frases já convertidas em IDs numéricos ou tokens
    lista_encoder_ids = []
    lista_decoder_ids = []
    
    print("Convertendo texto para matrizes de números (IDs)...")

    # 2. Iterando pelas nossas 1000 frases
    for item in dataset_filtrado['translation']:
        texto_en = item['en'] # Frases inglês
        texto_pt = item['pt'] # Frases Portugues 
        
        # Tokenizando o Inglês (Para o Encoder)
        # O parâmetro padding="max_length: server para preenchre a frase com zeros até bater o valor 32 da variavel MAX_LEN
        # O parâmetro truncation=True corta a frase se ela for maior que MAX_LEN
        tokens_en = tokenizer(
            texto_en, 
            max_length=MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt" # "pt" significa retornar tensores do PyTorch
        )
        
        # 3. Tokenizando o Português (Para o Decoder)
        # O BERT automaticamente embute o [CLS] (Start) no início e o [SEP] (EOS) no final
        # O parâmetro padding="max_length: server para preencher a frase com zeros até bater o valor 32 da variavel MAX_LEN
        # O parâmetro truncation=True corta a frase se ela for maior que MAX_LEN
        tokens_pt = tokenizer(
            texto_pt, 
            max_length=MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt" # "pt" significa retornar tensores do PyTorch
        )
        
        # Pegamos apenas a lista de IDs numéricos
        lista_encoder_ids.append(tokens_en['input_ids'])
        lista_decoder_ids.append(tokens_pt['input_ids'])
        
    # Agrupando as 1000 listas individuais em uma única Matriz Gigante 2D (1000 linhas x 32 colunas)
    matriz_encoder = torch.cat(lista_encoder_ids, dim=0)
    matriz_decoder = torch.cat(lista_decoder_ids, dim=0)
    
    print("Tokenização e Padding concluídos com sucesso!")
    print(f"Formato da Matriz de Entrada (Encoder): {matriz_encoder.shape}")
    print(f"Formato da Matriz Alvo (Decoder): {matriz_decoder.shape}")
    
    return matriz_encoder, matriz_decoder, tokenizer

# --- TESTANDO AS TAREFAS 1 E 2 JUNTAS ---
if __name__ == "__main__":
    #Supondo que você já rodou a função do passo anterior:
    meus_dados = carregar_dados_miniatura()
    matriz_x, matriz_y, meu_tokenizador = tokenizar_dataset(meus_dados)


# Tarefa 3: O Motor de Otimização (Training Loop)

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

# Modificação do Transformer do LAB04 de Numpy para Torch

class TransformerSimples(nn.Module):
    def __init__(self, tamanho_vocab, d_model=128):
        super().__init__()
        # Converte IDs em Vetores (Embeddings)
        self.embedding = nn.Embedding(tamanho_vocab, d_model)
        
        # Pesos do Encoder (W_q, W_k, W_v do Lab 04)
        self.W_q_enc = nn.Linear(d_model, d_model)
        self.W_k_enc = nn.Linear(d_model, d_model)
        self.W_v_enc = nn.Linear(d_model, d_model)
        
        # Pesos do Decoder (Masked Attention)
        self.W_q_dec1 = nn.Linear(d_model, d_model)
        self.W_k_dec1 = nn.Linear(d_model, d_model)
        self.W_v_dec1 = nn.Linear(d_model, d_model)
        
        # Pesos da Ponte (Cross-Attention)
        self.W_q_dec2 = nn.Linear(d_model, d_model)
        self.W_k_dec2 = nn.Linear(d_model, d_model)
        self.W_v_dec2 = nn.Linear(d_model, d_model)
        
        # Transformação Final para o tamanho do vocabulário
        self.transformacaoFinal = nn.Linear(d_model, tamanho_vocab)
        self.d_model = d_model

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Pega o tamanho da dimensão das chaves
        d_k = K.size(-1)

         # Multiplica as Perguntas (Q) pelas Chaves (K) para ver quais palavras combinam mais
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
         # Se tiver máscara (no Decoder), aplica ela escondendo as palavras futuras
        if mask is not None:
            # Substitui o 0 da máscara por -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Multiplica essas porcentagens pelos Valores (V) para criar o contexto final da palavra
        pesos = F.softmax(scores, dim=-1)
        return torch.matmul(pesos, V)

    def forward(self, x_enc, x_dec, mascara_causal):
        # Converte as listas de números inteiros em tensores com dimensões
        X = self.embedding(x_enc)
        Y = self.embedding(x_dec)
        
        # 2. Bloco do Encoder
        Q_enc = self.W_q_enc(X) # Cria a Pergunta
        K_enc = self.W_k_enc(X) # Cria a Chave
        V_enc = self.W_v_enc(X) # Cria o Valor
        Z = self.scaled_dot_product_attention(Q_enc, K_enc, V_enc) # Memória rica do Encoder
        
        # 3. BLoco Decoder Masked Self-Attention
        Q_dec1 = self.W_q_dec1(Y)
        K_dec1 = self.W_k_dec1(Y)
        V_dec1 = self.W_v_dec1(Y)
        Y_masked = self.scaled_dot_product_attention(Q_dec1, K_dec1, V_dec1, mask=mascara_causal)
        
        # 4. Bloco do Decoder Cross-Attention com Z do Encoder
        Q_dec2 = self.W_q_dec2(Y_masked)
        K_dec2 = self.W_k_dec2(Z)
        V_dec2 = self.W_v_dec2(Z)

        # Sem mascara 
        Saida_Decoder = self.scaled_dot_product_attention(Q_dec2, K_dec2, V_dec2) 
        
        # 5. Transformação Final
        logits = self.transformacaoFinal(Saida_Decoder)
        return logits


# Training Loop

def treinar_modelo(matriz_encoder, matriz_decoder, tamanho_vocab):
    print("\nIniciando a Tarefa 3: O Motor de Treinamento!\n")
    
    # 1. Instanciar o Modelo 
    modelo = TransformerSimples(tamanho_vocab, d_model=128)
    
    # 2. Definir a Função de Perda
    # ignore_index=0 diz para NÃO punir o modelo por errar os zeros do padding
    criterio_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    #3. Definir o Otimizador
    # O Parametro lr=0.001 é o tamanho do passo que ele dá a cada correção (Learning Rate)
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    # Criando uma máscara causal para o PyTorch 
    seq_len = matriz_decoder.size(1) - 1 # -1 porque vamos deslocar a frase
    mascara_causal = torch.tril(torch.ones((seq_len, seq_len))).bool()
    
    # 4. O Laço (Training Loop) - Rodar 15 vezes ou epocas
    epocas = 15
    
    for epoca in range(epocas):
        # Zera os gradientes da época anterior
        otimizador.zero_grad()
        
        # Entrada do Decoder: Pega tudo, menos a última palavra
        entrada_decoder = matriz_decoder[:, :-1] 
        # Saída Esperada: Pega tudo, menos a primeira palavra (<START>)
        alvo_esperado = matriz_decoder[:, 1:] 
        
        # Forward Pass: Passa os dados pelo modelo
        logits_previsoes = modelo(matriz_encoder, entrada_decoder, mascara_causal)
        
        # Remodelar as matrizes para a Função de Erro conseguir entender os números
        # O reshape(-1) alinha tudo em uma fila única gigante
        previsoes_achatadas = logits_previsoes.reshape(-1, tamanho_vocab) 
        alvos_achatados = alvo_esperado.reshape(-1)
        
        # Calcula o Erro (Loss), vai comparar as previsões com as respostas certas
        erro = criterio_loss(previsoes_achatadas, alvos_achatados)
        
        # Backward Pass: Calcula os gradientes
        erro.backward()
        
        # Atualiza os pesos
        otimizador.step()
        
        print(f"Época {epoca+1}/{epocas} | Loss (Erro): {erro.item():.4f}")
        
    print("\nTreinamento concluído! O Erro (Loss) diminuiu drasticamente!")
    return modelo

# Simulação apra testar o codigo da tarefa 3
if __name__ == "__main__":
    # Criando tensores falsos apenas para testar se o motor liga
    tam_vocab_falso = 5000
    matriz_enc_fake = torch.randint(1, 100, (10, 32))
    matriz_dec_fake = torch.randint(1, 100, (10, 32))
    
    # Rodar o modelo de treino
    modelo_treinado = treinar_modelo(matriz_enc_fake, matriz_dec_fake, tam_vocab_falso)








