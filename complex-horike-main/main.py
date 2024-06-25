import re
import nltk
from nltk.corpus import cmudict

# Ensure you have the necessary NLTK resources
nltk.download('cmudict')
nltk.download('words')

d = cmudict.dict()

# 1. Lungimea cuvântului
def lungimea_cuvantului(cuvant):
    return len(cuvant)

# 2. Numărul de silabe
def numar_silabe(cuvant):
    if cuvant.lower() in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[cuvant.lower()]][0]
    else:
        return 0

# 3. Prezența în limbajul cotidian
def prezenta_in_limbajul_cotidian(cuvant, corpus):
    return corpus.count(cuvant.lower())

# 4. Domeniul de specialitate
def domeniu_de_specialitate(cuvant, specialist_terms):
    return cuvant.lower() in specialist_terms

# 5. Originea lingvistică
def origine_lingvistica(cuvant, foreign_terms):
    return cuvant.lower() in foreign_terms

# 6. Complexitatea fonetică
def complexitate_fonetica(cuvant):
    if cuvant.lower() in d:
        pronunciations = d[cuvant.lower()]
        complexity = sum(len(pron) for pron in pronunciations) / len(pronunciations)
        return complexity
    else:
        return float('inf')  # Infinite complexity if not found

# Exemplu de utilizare
corpus = nltk.corpus.words.words()
specialist_terms = ["quantum", "algorithm", "neuroscience"]
foreign_terms = ["sushi", "fiesta", "schadenfreude"]

cuvant = "NAR"

print(f"Lungimea cuvântului: {lungimea_cuvantului(cuvant)}")
print(f"Numărul de silabe: {numar_silabe(cuvant)}")
print(f"Prezența în limbajul cotidian: {prezenta_in_limbajul_cotidian(cuvant, corpus)}")
print(f"Domeniul de specialitate: {domeniu_de_specialitate(cuvant, specialist_terms)}")
print(f"Originea lingvistică: {origine_lingvistica(cuvant, foreign_terms)}")
print(f"Complexitatea fonetică: {complexitate_fonetica(cuvant)}")
