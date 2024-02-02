import pandas as pd
import json
import math
from scipy.optimize import newton
import matplotlib.pyplot as plt

def geofonos():
    df = pd.read_pickle('dataframe.pkl')

    columnas_con_D = [col for col in df.columns if col.startswith('D')]
    numero_de_columnas_D = len(columnas_con_D) - 2
    columnas_micras = [f'D{i}' for i in range(1, numero_de_columnas_D + 1)]

    return numero_de_columnas_D


def tabla():

    df = pd.read_pickle('dataframe.pkl')

    columnas_con_D = [col for col in df.columns if col.startswith('D')]
    numero_de_columnas_D = len(columnas_con_D) - 2
    columnas_micras = [f'D{i}' for i in range(1, numero_de_columnas_D + 1)]

    numero_de_columnas_D = columnas_micras
    columnas_micras

    for col in columnas_micras:
        col_cm = col
        df[col_cm] = df[col] / 10000

    0.012596*(40 / 39.805952)
    df[col] * (40 / df['Carga'])

    cols_to_normalize = numero_de_columnas_D

    for col in columnas_micras:
        df[col] *= (40 / df['Carga'])
    #print(df)

    def calcular_FT(row, opcion):
        μ = -35.649 * (row['Espesor de capa de rodadura']) ** (-0.624)

        if opcion == 1:
            FT = 1 / (1 + (8 * (10 ** (-4)) * row['Espesor de capa de rodadura'] * (row['T-Man'] - 20)))
        elif opcion == 2:
            FT = (1.054) ** ((row['T-Man'] - 20) / μ)
        elif opcion == 3:
            FT = 1
        return FT

    def aplicar_opcion(df, opcion):
        df['FT'] = df.apply(lambda row: calcular_FT(row, opcion), axis=1)
        for columna in numero_de_columnas_D:
            df[columna] = df['FT'] * df[columna]
        return df

    opcion = 0

    with open('temperatura.txt', 'r') as file:
        opcion = file.read()
        opcion = int(opcion)

    if opcion == 1 or opcion == 2 or opcion == 3:
        df = aplicar_opcion(df, opcion)

    #Código para obtener las distancias de los geófonos
    valores_columna_d1 = df['D1']
    valores_columna_d1

    D = 90
    a = 22.5
    p = 0.2515

    #Obtener las columnas D y el índice de 'D2'
    columnas_D = sorted([col for col in df.columns if col.startswith('D')])
    indice_D2 = columnas_D.index('D2')

    #Crear una lista para almacenar las distancias de los geófonos
    distancias = []

    #Solicitar al usuario ingresar las distancias de los geófonos desde D2 hasta el penúltimo
    with open('distancia_D.txt', 'r') as file:
        contenido = file.read()
        distancias = [float(numero)* 2.54 for numero in contenido.split(',')]

    dataframe_unificado = pd.DataFrame()

    #Realizar cálculos y crear tablas separadas para cada columna D desde D2 hasta el penúltimo
    for i, columna_D in enumerate(columnas_D[indice_D2:-1], start=2):
        resultado = (2.4 * 40) / (df[columna_D] * distancias[i - 2])
        df_resultado = pd.DataFrame({f'Geofono seleccionado {columna_D}': resultado})
        df_resultado.drop(columns=['Modulo resiliente Distancia'], inplace=True, errors='ignore')
        dataframe_unificado = pd.concat([dataframe_unificado, df_resultado], axis=1)

    #Función para resolver la ecuación para un valor de d y Mr dado
    def solve_equation(d, Mr):
        def equation(E):
            return 1.5 * p * a * ((1 / (Mr * math.sqrt(1 + (D / a * (E / Mr) ** (2/3)) ** 2))) + (1 - 1 / math.sqrt(1 + (D / a) ** 2)) / E ) - d
        E_solution = newton(equation, x0=1)
        return E_solution

    #Función para calcular el número estructural
    def calcular_ne(D, E_solution):
        Ne = 0.02364 * D * (E_solution**(1/3))
        return Ne

    #Función para calcular el radio de tensiones del bulbo (ae)
    def calcular_ae(D, E, Mr):
        ae = math.sqrt(a ** 2 + (D * (E / Mr) ** (1/3)) ** 2)
        return ae

    #Inicializar el DataFrame geofonos_seleccionados
    geofonos_seleccionados = pd.DataFrame()

    #Iterar a través de dataframe_unificado
    for column_name, distancia in zip(dataframe_unificado.columns, distancias):
        Mr_values = dataframe_unificado[column_name]
        solutions = []

        distancia_column = df["Distancia"]

        for d_value, Mr in zip(valores_columna_d1, Mr_values):
            E_solution = solve_equation(d_value, Mr)
            ne_value = calcular_ne(D, E_solution)
            ae_value = calcular_ae(D, E_solution, Mr)

            #Verificación
            if distancia >= 0.7 * ae_value:
                cumple_condicion = 'Cumple'
            else:
                cumple_condicion = 'No Cumple'

            solutions.append((d_value, Mr, E_solution, ne_value, ae_value, cumple_condicion, distancia_column))

        if solutions:
            #DataFrame nuevo
            results_df = pd.DataFrame(solutions, columns=["D0", "Modulo resiliente", "Modulo de elasticidad", "Número estructural", "Radio de tensiones del bulbo (ae)", "Cumple Condición", "Absisado"])
            results_df["Absisado"] = distancia_column
            results_df['R'] = distancia

            #Agregar al DataFrame geofonos_seleccionados
            for absisado in results_df['Absisado'].unique():
                subset = results_df[results_df['Absisado'] == absisado]
                for _, row in subset.iterrows():
                    if row["Cumple Condición"] == "Cumple":
                        row_with_column = row.to_frame().T
                        row_with_column['Geofono seleccionado'] = column_name
                        geofonos_seleccionados = pd.concat([geofonos_seleccionados, row_with_column])

    #Reiniciar los índices del DataFrame resultante
    geofonos_seleccionados = geofonos_seleccionados.reset_index(drop=True)

    geofonos_seleccionados = geofonos_seleccionados.sort_values(by=["Absisado", "Geofono seleccionado"])
    geofonos_seleccionados = geofonos_seleccionados.drop_duplicates(subset=["Absisado"])

    columnas_a_eliminar = ["R", "D0", "Cumple Condición", "Radio de tensiones del bulbo (ae)"]
    geofonos_seleccionados = geofonos_seleccionados.drop(columns=columnas_a_eliminar)
    geofonos_seleccionados['Absisado'] = geofonos_seleccionados['Absisado'] / 1000
    geofonos_seleccionados['Diferencia'] = geofonos_seleccionados['Absisado'].diff()
    geofonos_seleccionados['Diferencia'] = geofonos_seleccionados['Diferencia'].fillna(0)
    geofonos_seleccionados['Distancia Acumulada'] = geofonos_seleccionados['Diferencia'].cumsum()

    geofonos_seleccionados['Promedio de Parámetro Modulo Resiliente'] = geofonos_seleccionados['Modulo resiliente'].rolling(window=2).mean()
    geofonos_seleccionados['Promedio de Parámetro Modulo Resiliente'] = geofonos_seleccionados['Promedio de Parámetro Modulo Resiliente'].fillna(geofonos_seleccionados['Modulo resiliente'])

    geofonos_seleccionados['Promedio de Parámetro Modulo de elasticidad'] = geofonos_seleccionados['Modulo de elasticidad'].rolling(window=2).mean()
    geofonos_seleccionados['Promedio de Parámetro Modulo de elasticidad'] = geofonos_seleccionados['Promedio de Parámetro Modulo de elasticidad'].fillna(geofonos_seleccionados['Modulo de elasticidad'])

    geofonos_seleccionados['Promedio de Parámetro Número estructural'] = geofonos_seleccionados['Número estructural'].rolling(window=2).mean()
    geofonos_seleccionados['Promedio de Parámetro Número estructural'] = geofonos_seleccionados['Promedio de Parámetro Número estructural'].fillna(geofonos_seleccionados['Número estructural'])

    geofonos_seleccionados['area del intervalo_mr'] = geofonos_seleccionados['Promedio de Parámetro Modulo Resiliente'] * geofonos_seleccionados['Diferencia']

    geofonos_seleccionados['area del intervalo Modulo de elasticidad'] = geofonos_seleccionados['Promedio de Parámetro Modulo de elasticidad'] * geofonos_seleccionados['Diferencia']

    geofonos_seleccionados['area del intervalo Número estructural'] = geofonos_seleccionados['Promedio de Parámetro Número estructural'] * geofonos_seleccionados['Diferencia']

    geofonos_seleccionados['Area acumulada_mr'] = geofonos_seleccionados['area del intervalo_mr'].cumsum()

    geofonos_seleccionados['Area acumulada Modulo de elasticidad'] = geofonos_seleccionados['area del intervalo Modulo de elasticidad'].cumsum()

    geofonos_seleccionados['Area acumulada Número estructural'] = geofonos_seleccionados['area del intervalo Número estructural'].cumsum()

    geofonos_seleccionados_copy = geofonos_seleccionados.copy()

    eliminar = ["Diferencia", "Distancia Acumulada", "Promedio de Parámetro Modulo Resiliente", "Promedio de Parámetro Modulo de elasticidad", "Promedio de Parámetro Número estructural", "area del intervalo_mr", "area del intervalo Modulo de elasticidad", "area del intervalo Número estructural", "Area acumulada_mr", "Area acumulada Modulo de elasticidad", "Area acumulada Número estructural"]
    geofonos_seleccionados_copy = geofonos_seleccionados_copy.drop(columns=eliminar)

    json_data = geofonos_seleccionados_copy.to_json(orient='records')

    ultimo_valor_area = geofonos_seleccionados['Area acumulada_mr'].iloc[-1]
    ultimo_valor_distancia = geofonos_seleccionados['Distancia Acumulada'].iloc[-1]
    geofonos_seleccionados['Zx'] = geofonos_seleccionados['Area acumulada_mr'] - (ultimo_valor_area / ultimo_valor_distancia) * geofonos_seleccionados['Distancia Acumulada']

    ultimo_valor_area_elasticidad = geofonos_seleccionados['Area acumulada Modulo de elasticidad'].iloc[-1]
    geofonos_seleccionados['Zy'] = geofonos_seleccionados['Area acumulada Modulo de elasticidad'] - (ultimo_valor_area_elasticidad / ultimo_valor_distancia) * geofonos_seleccionados['Distancia Acumulada']

    ultimo_valor_area_estructural = geofonos_seleccionados['Area acumulada Número estructural'].iloc[-1]
    geofonos_seleccionados['Zxy'] = geofonos_seleccionados['Area acumulada Número estructural'] - (ultimo_valor_area_estructural / ultimo_valor_distancia) * geofonos_seleccionados['Distancia Acumulada']

    geofonos_seleccionados

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Modulo resiliente'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Modulo Resiliente')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Modulo Resiliente')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image1.jpg')

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Zx'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Diferencias acumuladas Modulo Resiliente')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Modulo Resiliente')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image2.jpg')

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Número estructural'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Número estructural')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Número estructural')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image3.jpg')

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Zxy'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Diferencias acumuladas Modulo de elasticidad')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Modulo de elasticidad')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image4.jpg')

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Modulo de elasticidad'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Modulo de elasticidad')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Número estructural')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image5.jpg')

    plt.figure(figsize=(10, 6))
    plt.plot(geofonos_seleccionados['Absisado'], geofonos_seleccionados['Zy'], color='b', label='Datos', linestyle='-')
    plt.title('Gráfico de Absisado vs. Diferencias acumuladas Numero estructural')
    plt.xlabel('Absisado [km]')
    plt.ylabel('Numero estructural')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/image6.jpg')
    
    return json_data
