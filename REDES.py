import pandas as pd # manipulação de dataframes
import matplotlib.pyplot as plt # pacote de gráficos base
import seaborn as sns # criação de gráficos estatísticos atraentes e informativos para análise de dados
import os # operational system - permite navegar por diretórios

####################### GERAÇÃO DOS DATAFRAMES INDIVIDUAIS E O GERAL #######################

# Nomes dos arquivos CSV que vão ser gerados individualmente dado o otimizador empregado:
trainscg = pd.read_csv("trainscg.csv", delimiter=';')
trainlm = pd.read_csv("trainlm.csv", delimiter=';')
traincgb = pd.read_csv("traincgb.csv", delimiter=';')
trainbr = pd.read_csv("trainbr.csv", delimiter=';')
trainbfg = pd.read_csv("trainbfg.csv", delimiter=';')
traingdx = pd.read_csv("traingdx.csv", delimiter=';')
trainrp = pd.read_csv("trainrp.csv", delimiter=';')
trainoss = pd.read_csv("trainoss.csv", delimiter=';')

file_names = ["trainscg", "trainlm", "traincgb", "trainbr", "trainbfg", "traingdx", "trainrp", "trainoss"]
dataframes = {}

# Armazenamento de todos data frames individuais em somente um:
for file_name in file_names:
    df = pd.read_csv(f"{file_name}.csv", delimiter=';')
    dataframes[file_name] = df

tabela_final = pd.concat(dataframes.values(), ignore_index=True)

# Ordenar a tabela_final pela coluna 37 (MAPE):
tabela_ordenada = tabela_final.sort_values(tabela_final.columns[37])
dez_menores = tabela_ordenada.head(10) # Selecionar os 10 primeiros registros
TOP_10_RESULTADOS = dez_menores.iloc[:, [2, 3, 4, 5, 6, 7, 37]] # colunas desejadas

# quem vai ser plotado. salvos TODOS serão! - cuidado com os diretórios.
mostrar = 4 # aqui pode ser colocado 1, 2, 3, 4 ou "todos" conforme a análise que você desejar.


####################### GRÁFICOS DE ANÁLISE NEURÔNIO (1ª CAMADA) x ERRO #######################

# DESCRIÇÃO1: aqui, será feita uma análise do MAPE gerado a partir de um dado número de neurônio na 1ª e sua respectiva função de ativação.

# diretório 1 (o 'r' na frente é de raw (cru do inglês), para burlar os caracteres de escape que possuem barra invertida como \n, \t, \\, \' ou \")
SAVE1 = r'C:\Users\orlan\Downloads\TCC 1\MATLAB\Matlab\2 Camada - regressão\2 Camada - regressão\ALGORITMOS\ESTRATÉGIA 3 - PRIME\CSVs\SAVE1'

# Defina uma paleta de cores personalizada
palette = sns.color_palette("Set1", n_colors=len(dataframes))

for file_name, df in dataframes.items():
    plt.figure(figsize=(12, 8))
    plt.ylim(0.15, 10)  # Limitando os valores do eixo Y

    # Agrupando por função de ativação da primeira camada e algoritmo de otimização
    grupos = df.groupby([df.columns[3]])
    
    # Iterando sobre cada grupo para plotar os pontos e as linhas
    for i, (nome, grupo) in enumerate(grupos):
        # Use a cor da paleta de cores personalizada para cada grupo
        color = palette[i]
        
        # Escurecendo a cor principal para obter a cor do contorno
        contour_color = sns.set_hls_values(color, l=0.3)
        
        # Scatterplot com cores personalizadas
        sns.scatterplot(x=grupo.iloc[:, 6], y=grupo.iloc[:, 37], label=nome, color=color, edgecolor=contour_color, linewidth=0.5)
        
    # Ajustando detalhes do gráfico
    plt.title(f'MAPE em Função do Número de Neurônios da Primeira Camada ({file_name})')
    plt.xlabel('Número de Neurônios da Primeira Camada')
    plt.ylabel('MAPE')
    plt.legend(title='Função de Ativação')

    # Definindo o intervalo do eixo X para mostrar valores de 5 em 5
    x_min = df.iloc[:, 6].min()
    x_max = df.iloc[:, 6].max()
    plt.xticks(range(int(x_min), int(x_max) + 1, 5))

    CAMINHO1 = os.path.join(SAVE1, f"{file_name}_1_CAMADA.png") # Salvando a figura no diretório especificado
    plt.savefig(CAMINHO1)
    if mostrar == 1 or mostrar == "todos":
        plt.show()  # Fecha a figura atual para não exibir na saída
    else:
        plt.close()



####################### GRÁFICOS DE ANÁLISE NEURÔNIO (2ª CAMADA) x ERRO #######################

# DESCRIÇÃO2: aqui, será feita uma análise do MAPE gerado a partir de um dado número de neurônio na 2ª e sua respectiva função de ativação.

# diretório 2
SAVE2 = r'C:\Users\orlan\Downloads\TCC 1\MATLAB\Matlab\2 Camada - regressão\2 Camada - regressão\ALGORITMOS\ESTRATÉGIA 3 - PRIME\CSVs\SAVE2'

# Defina uma paleta de cores personalizada
palette = sns.color_palette("Set1", n_colors=len(file_names))

for file_name in file_names:
    df = pd.read_csv(f"{file_name}.csv", delimiter=';')
    # Convertendo colunas para o tipo correto
    df.iloc[:, 7] = pd.to_numeric(df.iloc[:, 7], errors='coerce')  # Coluna 7: Número de neurônios da 2ª camada
    df.iloc[:, 37] = pd.to_numeric(df.iloc[:, 37], errors='coerce') # Coluna 37: MAPE
    dataframes[file_name] = df

# Criando um gráfico para cada dataframe
for file_name, df in dataframes.items():
    plt.figure(figsize=(12, 8))
    plt.ylim(.15, 100)  # Limitando os valores do eixo Y
    # Agrupando por função de ativação da segunda camada e algoritmo de otimização
    grupos = df.groupby([df.columns[4]])
    # Iterando sobre cada grupo para plotar os pontos e as linhas
    for i, (nome, grupo) in enumerate(grupos):
        # Use a cor da paleta de cores personalizada para cada grupo
        color = palette[i]
        
        # Obtenha uma cor mais escura para o contorno
        contour_color = sns.dark_palette(color, n_colors=1, input="rgb")[0]
        
        # Scatterplot com cores personalizadas
        sns.scatterplot(x=grupo.iloc[:, 7], y=grupo.iloc[:, 37], label=nome, color=color, edgecolor=contour_color, linewidth=0.5)
        
    # Ajustando detalhes do gráfico
    plt.title(f'MAPE em Função do Número de Neurônios da Segunda Camada ({file_name})')
    plt.xlabel('Número de Neurônios da Segunda Camada')
    plt.ylabel('MAPE')
    plt.legend(title='Função de Ativação')

    # Definindo o intervalo do eixo X para mostrar valores de 5 em 5
    x_min = df.iloc[:, 7].min()
    x_max = df.iloc[:, 7].max()
    plt.xticks(range(int(x_min), int(x_max) + 1, 5))

    # Salvando a figura
    CAMINHO2 = os.path.join(SAVE2, f"{file_name}_2_CAMADA.png") # Salvando a figura no diretório especificado
    plt.savefig(CAMINHO2)
    if mostrar == 2 or mostrar == "todos":
        plt.show()  # Fecha a figura atual para não exibir na saída
    else:
        plt.close()

        
####################### GRÁFICOS DE ANÁLISE NEURÔNIO (SAÍDA) x ERRO #######################

# DESCRIÇÃO3: aqui, será feita uma análise do MAPE gerado a partir de um dado número de neurônio na 2ª camada dado uma função de ativação na saída.

# diretório 3
SAVE3 = r'C:\Users\orlan\Downloads\TCC 1\MATLAB\Matlab\2 Camada - regressão\2 Camada - regressão\ALGORITMOS\ESTRATÉGIA 3 - PRIME\CSVs\SAVE3'

# Defina uma paleta de cores personalizada
palette = sns.color_palette("Set1", n_colors=len(file_names))

for file_name in file_names:
    df = pd.read_csv(f"{file_name}.csv", delimiter=';')
    # Convertendo colunas para o tipo correto
    df.iloc[:, 7] = pd.to_numeric(df.iloc[:, 7], errors='coerce')  # Coluna 7: Número de neurônios da saída
    df.iloc[:, 37] = pd.to_numeric(df.iloc[:, 37], errors='coerce') # Coluna 37: MAPE
    dataframes[file_name] = df

# Criando um gráfico para cada dataframe
for file_name, df in dataframes.items():
    plt.figure(figsize=(12, 8))
    plt.ylim(2, 2.1)  # Limitando os valores do eixo Y
    # Agrupando por função de ativação da saída e algoritmo de otimização
    grupos = df.groupby([df.columns[5]])
    # Iterando sobre cada grupo para plotar os pontos e as linhas
    for i, (nome, grupo) in enumerate(grupos):
        # Use a cor da paleta de cores personalizada para cada grupo
        color = palette[i]
        
        # Obtenha uma cor mais escura para o contorno
        contour_color = sns.dark_palette(color, n_colors=1, input="rgb")[0]
        
        # Scatterplot com cores personalizadas
        sns.scatterplot(x=grupo.iloc[:, 7], y=grupo.iloc[:, 37], label=nome, color=color, edgecolor=contour_color, linewidth=0.5)
        
    # Ajustando detalhes do gráfico
    plt.title(f'MAPE em Função do Número de Neurônios da 2ª camada pela função na saída ({file_name})')
    plt.xlabel('Número de Neurônios na 2ª camada')
    plt.ylabel('MAPE')
    plt.legend(title='Função de Ativação')

    # Definindo o intervalo do eixo X para mostrar valores de 5 em 5
    x_min = df.iloc[:, 7].min()
    x_max = df.iloc[:, 7].max()
    plt.xticks(range(int(x_min), int(x_max) + 1, 5))

    CAMINHO3 = os.path.join(SAVE3, f"{file_name}_SAÍDA.png") # Salvando a figura no diretório especificado
    plt.savefig(CAMINHO3)
    if mostrar == 3 or mostrar == "todos":
        plt.show()  # Fecha a figura atual para não exibir na saída
    else:
        plt.close()
    
################## GRÁFICOS DE ANÁLISE NEURÔNIOS DA 1ª E 2ª CAMADAS x ERRO #######################

# DESCRIÇÃO4: aqui, será feita uma análise do MAPE gerado a partir de um dado número de neurônio na 2ª fixando um valor da 1ª camada.

# diretório 4
SAVE4 = r'C:\Users\orlan\Downloads\TCC 1\MATLAB\Matlab\2 Camada - regressão\2 Camada - regressão\ALGORITMOS\ESTRATÉGIA 3 - PRIME\CSVs\SAVE4'

# Escolha a cor desejada para o gráfico
cor_do_grafico = 'orange'  # Substitua pela cor que desejar

for file_name, df in dataframes.items():
    # Convertendo colunas para o tipo correto e removendo valores não numéricos
    df.iloc[:, 6] = pd.to_numeric(df.iloc[:, 6], errors='coerce')  # Coluna 6: Número de neurônios da primeira camada
    df.iloc[:, 7] = pd.to_numeric(df.iloc[:, 7], errors='coerce')  # Coluna 7: Número de neurônios da saída
    df.iloc[:, 37] = pd.to_numeric(df.iloc[:, 37], errors='coerce') # Coluna 38: MAPE
    df.dropna(subset=[df.columns[6], df.columns[7], df.columns[37]], inplace=True)

    # Encontre o valor mínimo do eixo X
    x_min = df.iloc[:, 7].min()

    # Iterando sobre cada valor único da coluna 6
    for valor_col6 in df.iloc[:, 6].unique():
        subset_df = df[df.iloc[:, 6] == valor_col6]

        # Filtrando para mostrar apenas múltiplos de 5 no eixo X
        subset_df = subset_df[subset_df.iloc[:, 7] % 5 == 0]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=subset_df.iloc[:, 7], y=subset_df.iloc[:, 37], color=cor_do_grafico)  # Use a cor definida
        
        plt.title(f'MAPE em Função do Número de Neurônios da Saída ({file_name}) - Neurônios Primeira Camada: {valor_col6}')
        plt.xlabel('Número de Neurônios da Saída')
        plt.ylabel('MAPE')

        # Definindo o limite do eixo Y
        plt.ylim(.01, 10)
        # Ajustando a escala do eixo X para múltiplos de 5
        plt.xticks(range(int(x_min), int(subset_df.iloc[:, 7].max() + 1), 5))

        CAMINHO4 = os.path.join(SAVE4, f"{file_name}_saida_{valor_col6}.png") # Salvando a figura no diretório especificado
        plt.savefig(CAMINHO4)
        if mostrar == 4 or mostrar == "todos":
            plt.show()  # Fecha a figura atual para não exibir na saída
        else:
            plt.close()
            
###################################### EXPORTAÇÃO DA TABELA COMPLETA PARA O EXCEL #################################################

# Especifique o caminho completo do arquivo Excel no qual deseja salvar o DataFrame
caminho_arquivo_excel = r'C:\Users\orlan\Downloads\TCC 1\MATLAB\Matlab\2 Camada - regressão\2 Camada - regressão\ALGORITMOS\ESTRATÉGIA 3 - PRIME\CSVs\tabela_final.xlsx'

# Use a função to_excel para exportar o DataFrame para um arquivo Excel
tabela_final.to_excel(caminho_arquivo_excel, index=True)  # Defina index=False para não incluir o índice no arquivo Excel




