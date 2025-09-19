from services.bncc_analysis_clean import process_uploaded_file

if __name__ == "__main__":
    # Exemplo de parâmetros (ajuste conforme seus arquivos reais)
    curriculo_file = "data/curriculo/curriculo_anos_finais_orientacoes.xlsx"
    segment = "anos_finais"
    nota_corte = 0.8

    print("Testando process_uploaded_file...")
    resultado = process_uploaded_file(curriculo_file, segment, nota_corte)
    print("Resultado:")
    print(resultado)
