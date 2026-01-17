import PyPDF2

pdf_path = "/media/pc_remi/Synchro/ESISAR/3_Semestre/IN451/Bonbons/Automatic_candy_selector/Automatic-Candy-Selector/IN450_451-Sujet_Projet_IA_MachineLearningV2_5.pdf"

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Nombre de pages: {len(reader.pages)}\n")
        print("="*80)
        
        for i, page in enumerate(reader.pages):
            print(f"\n--- PAGE {i+1} ---\n")
            print(page.extract_text())
            print("="*80)
except Exception as e:
    print(f"Erreur: {e}")
