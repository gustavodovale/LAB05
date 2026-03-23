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
    
    modelo = TransformerSimples(tamanho_vocab, d_model=128)
    criterio_loss = nn.CrossEntropyLoss(ignore_index=0)
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    # Vamos rodar poucas épocas (ex: 5) só para mostrar o Loss caindo
    epocas = 5
    batch_size = 32 # <-- A MÁGICA PARA NÃO TRAVAR A MEMÓRIA
    
    for epoca in range(epocas):
        # Zera o acumulador de erros toda vez que uma nova época (rodada de estudos) começa
        erro_total_epoca = 0 
        
        # --- O LAÇO DOS MINI-BATCHES (A MÁGICA PARA NÃO TRAVAR O PC) ---
        # Em vez de ler as 1000 frases de uma vez, pulamos de 32 em 32 (batch_size)
        for i in range(0, len(matriz_encoder), batch_size):
            
            # 1. Pega apenas a "fatia" atual de 32 frases do Encoder e do Decoder
            batch_enc = matriz_encoder[i:i+batch_size]
            batch_dec = matriz_decoder[i:i+batch_size]
            
            # 2. Limpa o lixo de memória (gradientes) do cálculo da fatia anterior
            otimizador.zero_grad()
            
            # 3. O Truque do Deslocamento (Teacher Forcing)
            # Entrada do Decoder: Pega a frase inteira, mas joga fora a última palavra
            entrada_decoder = batch_dec[:, :-1] 
            # Saída Esperada (Gabarito): Pega a frase inteira, mas joga fora a primeira palavra (<START>)
            alvo_esperado = batch_dec[:, 1:] 
            
            # 4. Cria a máscara para o tamanho atual da frase (impede o modelo de olhar o futuro)
            seq_len = entrada_decoder.size(1)
            mascara_causal = torch.tril(torch.ones((seq_len, seq_len))).bool()
            
            # 5. FORWARD PASS (O Chute): Passa as 32 frases pelo modelo para ver o que ele adivinha
            logits_previsoes = modelo(batch_enc, entrada_decoder, mascara_causal)
            
            # 6. Achata a matriz 3D para uma fila única para o "professor" (Loss) conseguir corrigir
            previsoes_achatadas = logits_previsoes.reshape(-1, tamanho_vocab) 
            alvos_achatados = alvo_esperado.reshape(-1)
            
            # 7. Calcula o Erro comparando as adivinhações com o gabarito
            erro = criterio_loss(previsoes_achatadas, alvos_achatados)
            
            # 8. BACKWARD PASS: Calcula onde o modelo errou usando derivadas (A mágica do PyTorch)
            erro.backward()
            
            # 9. Atualiza as matrizes de peso para ele ficar mais inteligente na próxima rodada
            otimizador.step()
            
            # 10. Guarda o erro dessa fatia para podermos calcular a média no final
            erro_total_epoca += erro.item()
            
        # --- FIM DA ÉPOCA ---
        # Calcula a média do erro pegando a soma total e dividindo pela quantidade de fatias que fizemos
        qtd_batches = len(matriz_encoder) / batch_size
        erro_medio = erro_total_epoca / qtd_batches
        
        # Mostra na tela como o erro está caindo!
        print(f"Época {epoca+1}/{epocas} | Loss (Erro Médio): {erro_medio:.4f}")
        
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


# --- TESTANDO AS TAREFAS 1 2, 3, 4 JUNTAS ---
if __name__ == "__main__":

    meus_dados = carregar_dados_filtrado()
    matriz_x, matriz_y, meu_tokenizador = tokenizar_dataset(meus_dados)

    # 3. Pega o tamanho REAL do vocabulário do tokenizador do BERT
    tamanho_vocab_real = meu_tokenizador.vocab_size

    modelo_treinado = treinar_modelo(matriz_x, matriz_y, tamanho_vocab_real)

    # 4. Overfitting Test
    # Pega a primeira frase que ele memorizou
    frase_teste = meus_dados[0]['translation']['en']
    traducao = traduzir_frase(modelo_treinado, meu_tokenizador, frase_teste)






