# Bibliotecas Necessarias
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F


# Tarefa 1: Preparando o Dataset

def carregar_dados_filtrado():
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
    #meus_dados = carregar_dados_filtrado()
    
    # teste de funcionamento dos dados carregados
    # exemplo = meus_dados[0]['translation']
    # print("Exemplo de dado carregado:")
    # print(f"🇺🇸 Inglês: {exemplo['en']}")
    # print(f"🇧🇷 Português: {exemplo['pt']}")


# Tarefa 2: Tokenização Básica

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
    
    print("Tokenização e Padding concluídos com sucesso!\n")
    print(f"Formato da Matriz de Entrada (Encoder): {matriz_encoder.shape}\n")
    print(f"\nFormato da Matriz Alvo (Decoder): {matriz_decoder.shape}\n")
    
    return matriz_encoder, matriz_decoder, tokenizer

# # --- TESTANDO AS TAREFAS 1 E 2 JUNTAS ---
# if __name__ == "__main__":
#     #Supondo que você já rodou a função do passo anterior:
#     meus_dados = carregar_dados_filtrado()
#     matriz_x, matriz_y, meu_tokenizador = tokenizar_dataset(meus_dados)


# Tarefa 3: O Motor de Otimização (Training Loop)

# Modificação do Transformer do LAB04 de Numpy para Torch

class TransformerSimples(nn.Module):
    def __init__(self, tamanho_vocab, d_model=128, max_len=32):
        super().__init__()
        self.embedding = nn.Embedding(tamanho_vocab, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.W_q_enc = nn.Linear(d_model, d_model)
        self.W_k_enc = nn.Linear(d_model, d_model)
        self.W_v_enc = nn.Linear(d_model, d_model)
        
        self.W_q_dec1 = nn.Linear(d_model, d_model)
        self.W_k_dec1 = nn.Linear(d_model, d_model)
        self.W_v_dec1 = nn.Linear(d_model, d_model)
        
        self.W_q_dec2 = nn.Linear(d_model, d_model)
        self.W_k_dec2 = nn.Linear(d_model, d_model)
        self.W_v_dec2 = nn.Linear(d_model, d_model)
        
        self.ffn1 = nn.Linear(d_model, d_model * 2)
        self.ffn2 = nn.Linear(d_model * 2, d_model)
        
        self.transformacaoFinal = nn.Linear(d_model, tamanho_vocab)
        self.d_model = d_model

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        pesos = F.softmax(scores, dim=-1)
        return torch.matmul(pesos, V)

    def forward(self, x_enc, x_dec, mascara_causal):
        seq_len_enc = x_enc.size(1)
        seq_len_dec = x_dec.size(1)
        posicoes_enc = torch.arange(0, seq_len_enc).unsqueeze(0)
        posicoes_dec = torch.arange(0, seq_len_dec).unsqueeze(0)
        
        X = self.embedding(x_enc) + self.pos_embedding(posicoes_enc)
        Y = self.embedding(x_dec) + self.pos_embedding(posicoes_dec)
        
        # --- ENCODER ---
        Q_enc = self.W_q_enc(X) 
        K_enc = self.W_k_enc(X) 
        V_enc = self.W_v_enc(X) 
        # A MÁGICA 1: Somando X (Conexão Residual) para curar a amnésia!
        Z = X + self.scaled_dot_product_attention(Q_enc, K_enc, V_enc) 
        
        # --- DECODER (Self-Attention) ---
        Q_dec1 = self.W_q_dec1(Y)
        K_dec1 = self.W_k_dec1(Y)
        V_dec1 = self.W_v_dec1(Y)
        # A MÁGICA 2: Somando Y
        Y_masked = Y + self.scaled_dot_product_attention(Q_dec1, K_dec1, V_dec1, mask=mascara_causal)
        
        # --- DECODER (Cross-Attention) ---
        Q_dec2 = self.W_q_dec2(Y_masked)
        K_dec2 = self.W_k_dec2(Z)
        V_dec2 = self.W_v_dec2(Z)
        # A MÁGICA 3: Somando Y_masked
        Saida_Decoder = Y_masked + self.scaled_dot_product_attention(Q_dec2, K_dec2, V_dec2) 
        
        # Cérebro Lógico com Conexão Residual Final
        saida_ffn = self.ffn2(F.relu(self.ffn1(Saida_Decoder)))
        saida_final = Saida_Decoder + saida_ffn
        
        logits = self.transformacaoFinal(saida_final)
        return logits

# Training Loop

def treinar_modelo(matriz_encoder, matriz_decoder, tamanho_vocab):
    print("\nIniciando a Tarefa 3: O Motor de Treinamento (Modo Memorização)!\n")
    
    modelo = TransformerSimples(tamanho_vocab, d_model=128)
    criterio_loss = nn.CrossEntropyLoss(ignore_index=0)
    # Aumentei o passo (lr) de leve para ele aprender mais rápido
    otimizador = optim.Adam(modelo.parameters(), lr=0.001) 
    
    epocas = 200
    batch_size = 1 # Vamos passar 1 frase por vez
    
    for epoca in range(epocas):
        erro_total_epoca = 0 
        
        for i in range(0, len(matriz_encoder), batch_size):
            batch_enc = matriz_encoder[i:i+batch_size]
            batch_dec = matriz_decoder[i:i+batch_size]
            
            otimizador.zero_grad()
            
            entrada_decoder = batch_dec[:, :-1] 
            alvo_esperado = batch_dec[:, 1:] 
            
            seq_len = entrada_decoder.size(1)
            mascara_causal = torch.tril(torch.ones((seq_len, seq_len))).bool()
            
            logits_previsoes = modelo(batch_enc, entrada_decoder, mascara_causal)
            
            previsoes_achatadas = logits_previsoes.reshape(-1, tamanho_vocab) 
            alvos_achatados = alvo_esperado.reshape(-1)
            
            erro = criterio_loss(previsoes_achatadas, alvos_achatados)
            erro.backward()
            otimizador.step()
            
            erro_total_epoca += erro.item()
            
        qtd_batches = len(matriz_encoder) / batch_size
        erro_medio = erro_total_epoca / qtd_batches
        
        # Mostra o Loss caindo a cada 20 épocas para não poluir a tela
        if (epoca + 1) % 20 == 0 or epoca == 0:
            print(f"Época {epoca+1:03d}/{epocas} | Loss (Erro Médio): {erro_medio:.4f}")
        
    print("\nTreinamento concluído! O Erro (Loss) diminuiu!")
    return modelo

# # Simulação apra testar o codigo da tarefa 3
# if __name__ == "__main__":
#     # Criando tensores falsos apenas para testar se o motor liga
#     tam_vocab_falso = 5000
#     matriz_enc_fake = torch.randint(1, 100, (10, 32))
#     matriz_dec_fake = torch.randint(1, 100, (10, 32))
    
#     # Rodar o modelo de treino
#     modelo_treinado = treinar_modelo(matriz_enc_fake, matriz_dec_fake, tam_vocab_falso)


# Tarefa 4: A Prova de Fogo (Overfitting Test)

def traduzir_frase(modelo, tokenizador, frase_original, max_len=32):
    print(f"\n--- TAREFA 4: A Prova de Fogo (Overfitting Test) ---")
    print(f"Frase Original (Inglês): {frase_original}")
    
    # 1. Modo de Avaliação: Avisamos ao PyTorch que o treino acabou.
    # Isso desliga os cálculos de gradientes e otimização para o modelo rodar mais rápido.
    modelo.eval()
    
    # 2. Transforma a frase de texto (Inglês) em números (IDs) para o Encoder ler
    tokens_entrada = tokenizador(
        frase_original, 
        max_length=max_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    # x_enc é a matriz de entrada do Encoder
    x_enc = tokens_entrada['input_ids']
    
    # 3. Prepara a entrada do Decoder começando APENAS com o token de início <START>
    # No dicionário do BERT, o token de início [CLS] é o número 101
    id_start = tokenizador.cls_token_id 
    # E o token de fim [SEP] ou <EOS> é o número 102
    id_eos = tokenizador.sep_token_id   
    
    # Cria uma matriz pequena para o Decoder que vai crescer a cada palavra adivinhada
    x_dec = torch.tensor([[id_start]])
    
    print("Iniciando a geração palavra por palavra...")
    
    # 4. O Laço Auto-Regressivo (O motor de adivinhação)
    with torch.no_grad(): 
        for passo in range(max_len):
            # Cria a máscara causal para o tamanho atual da frase do Decoder
            # Isso impede o Decoder de tentar roubar a resposta do futuro
            seq_len_atual = x_dec.size(1)
            mascara_causal = torch.tril(torch.ones((seq_len_atual, seq_len_atual))).bool()
            
            # O modelo olha para o Inglês (x_enc) e para o que já traduziu (x_dec) e tenta adivinhar o resto
            logits = modelo(x_enc, x_dec, mascara_causal)
            
            # Pega as pontuações (chances) apenas da ÚLTIMA palavra gerada na fila
            # [0] = primeira frase, [-1] = última palavra, [:] = todas as probabilidades do dicionário
            probabilidades_ultima_palavra = logits[0, -1, :]
            
            # Escolhe a palavra com a maior pontuação (a grande vencedora do Softmax)
            id_vencedor = torch.argmax(probabilidades_ultima_palavra).unsqueeze(0).unsqueeze(0)
            
            # Junta a palavra nova na frase do Decoder para a próxima rodada (Feedback Loop)
            x_dec = torch.cat([x_dec, id_vencedor], dim=1)
            
            # Pega a palavra em texto puro só para mostrar na tela o que ele está pensando
            palavra_texto = tokenizador.decode(id_vencedor[0])
            print(f"Passo {passo+1}: O modelo escolheu -> {palavra_texto}")
            
            # Se ele cuspir o token de fim <EOS> (id 102), a frase acabou e podemos parar o laço!
            if id_vencedor.item() == id_eos:
                break
    
    # 5. Converte todos os IDs gerados de volta para uma frase legível em Português
    # O comando skip_special_tokens=True limpa a sujeira visual tirando o [CLS], [SEP] e os zeros da tela
    traducao_final = tokenizador.decode(x_dec[0], skip_special_tokens=True)
    
    print("\nResultado Final da Prova de Fogo:")
    print(f"Tradução Gerada: {traducao_final}")
    
    return traducao_final


# #--- TESTANDO AS TAREFAS 1 2, 3, 4 JUNTAS ---
# if __name__ == "__main__":

#     meus_dados = carregar_dados_filtrado()
#     matriz_x, matriz_y, meu_tokenizador = tokenizar_dataset(meus_dados)

#     # 3. Pega o tamanho REAL do vocabulário do tokenizador do BERT
#     tamanho_vocab_real = meu_tokenizador.vocab_size

#     matriz_x_pequena = matriz_x[:5] # Pegar só 5 frases
#     matriz_y_pequena = matriz_y[:5]

#     #modelo_treinado = treinar_modelo(matriz_x, matriz_y, tamanho_vocab_real)
#     modelo_treinado = treinar_modelo(matriz_x_pequena, matriz_y_pequena, tamanho_vocab_real)

#     # 4. Overfitting Test
#     # Pega a primeira frase que ele memorizou
#     frase_teste = meus_dados[0]['translation']['en']
#     traducao = traduzir_frase(modelo_treinado, meu_tokenizador, frase_teste)


# --- TESTANDO AS TAREFAS 1 2, 3, 4 JUNTAS ---
if __name__ == "__main__":

    meus_dados = carregar_dados_filtrado()
    matriz_x, matriz_y, meu_tokenizador = tokenizar_dataset(meus_dados)

    # 3. Pega o tamanho REAL do vocabulário do tokenizador do BERT
    tamanho_vocab_real = meu_tokenizador.vocab_size

    # --- ISOLANDO 1 ÚNICA FRASE PARA O TESTE ---
    matriz_x_pequena = matriz_x[:1] # Pegar SÓ a frase 1
    matriz_y_pequena = matriz_y[:1] # Pegar SÓ o gabarito 1

    modelo_treinado = treinar_modelo(matriz_x_pequena, matriz_y_pequena, tamanho_vocab_real)

    # 4. Overfitting Test
    # Pega a primeira frase que ele memorizou exaustivamente
    frase_teste = meus_dados[0]['translation']['en']
    traducao = traduzir_frase(modelo_treinado, meu_tokenizador, frase_teste)





