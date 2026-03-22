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


    











