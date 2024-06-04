[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/bqhyehlj)
# Predicția complexității cuvintelor

În acest repo trebuie să puneți:
- codul împreună cu dependințele ce trebuie executate pentru a produce rezultatul de pe leaderbord
- un raport pdf detaliat mai jos


## Cuprins

- [Cerință și notare](#note)
- [Ce este complexitatea unui cuvânt?](#complex)
- [Caracteristici](#features)
- [Cod de start](#start)
- [Resurse permize](#resurse)
- [Sugestii de caracteristici](#sugestii)
- [Evaluarea modelelor](#eval)
- [Despre date](#data)
- [Model de submission](#submission)

<a name="note"></a> 
## Cerință și notare
- trebuie să antrenați modele de regresie care să prezică complexitatea cuvintelor pe datele de test
- proiectul valorează 20% din nota finală și nu se poate recupera la restanță sau la mărire decât prin rezolvarea de exerciții suplimentare
- echipe de maxim **2 persoane**, notarea se face individual pe baza întrebărilor verbale

**Nota pe proiect se ia în funcție de 3 criterii:**
1. 50% rezultat în clasament cu cod reproductibil: script gata executabil sau notebook colab sau notebook kaggle 
2. 50% un raport 1-2 pagini în română sau engleză despre ce ați încercat, ce a mers, ce nu, exploratory data analysis; folosiți template-ul de aici: https://www.overleaf.com/read/rkdqmfsgbjrm#042599
3. discuție verbală despre rezultate după examenul de pe 25 iunie


### Barem:
- toți ce care depășesc baseline-ul primesc nota 6 pe cod cu condiția ca codul să fie executabil
- cine nu depășește baseline-ul primește 5 dacă face o analiză exploratorie a datelor în raport
- raportul trebuie scris într-un [stil academic formal](https://subjectguides.york.ac.uk/academic-writing/academic-style), poate fi în română sau engleză dar fără LLM; rapoartele care conțin multe cuvinte și nu spun nimic (generate cu LLM) primesc 0
- orice model folosit trebuie explicat verbal sau în scris; modelele pe care nu le puteți explica se depunctează sau se punctează cu 0
- locul în clasamentul final (private leaderboard) va influența nota cu câte o sutime:

|poziție|  nota |
|------:|------:|
|     1 |  10   |
|     2 |  9.9  |
|     3 |  9.8  |
|     4 |  9.7  |
|     5 |  9.6  |
|     6 |  9.5  |
|     7 |  9.4  |
|     8 |  9.3  |
|     9 |  9.2  |
|     10 | 9.1  |
|     11 | 9    |
|     .. | ..    |


<a name="complex"></a> 
## Ce este complexitatea unui cuvânt?

Complexitatea unui cuvânt este un criteriu subiectiv și depinde de mulți factori de la cât de des este întâlnit în vorbire acel cuvânt, cât de lung sau greu de citit este, dacă este un termen specializat, forma morfologică, semantică până la funcția cuvântului în sintaxa propoziției.  Pe baza acestor idei, putem să ne definim niște funcții care să extragă diverse caracterisitici.


<a name="features"></a> 
## Caracteristici
Caracteristicile pentru un cuvânt sunt sub forma unui vector:
$$\mathbf{x} = \{ x_1, \cdots, x_n \}$$

unde fiecare componentă $x_i$ e o valoare numerică care poate fi extrasă prin euristici:

- cu privire la cuvânt: frecvență într-un corpus de dimensiuni mari, frecvență într-un corpus de texte pentru copii, lungime, nr de silabe, nr de consoane, dacă e abreviere
- cu privire la sintaxă: parte de vorbire, funcția în propozție
- caracteristici psiholingvistice: vârsta de achiziție, imaginabilitatea cuvântului
- caracteristici semantice: număr de sinonime, antonime, cuvinte similare din alte propoziții
- caracteristici binare: prezența în anumite liste externe, dacă e de tip titlu, dacă este entitate numită etc
 
Pentru fiecare astfel de vector vrem să prezicem o valoare numerică y care reprezintă scorul de complexitate.
Scorul de complexitate depinde de mulți factori, nu doar de proprietățile individuale ale cuvântului, ci și de contextul în care este folosit, funcția și sensurile pe care le are în contextul respectiv.

<a name="start"></a> 
## Cod de start
În repo aveți câteva notebooks de la care puteți porni. Acestea pot fi uploadate și pe kaggle și pot rula direct acolo (dar trebuie să elimnați prima celulă care downloadează datele).

- baseline.ipynb, [colab](https://colab.research.google.com/drive/1AhovRCbL5jyLEzAS9TFgu7igILJF6dCm?usp=sharing) - vă arată un proces de extragere a caracteristicilor din text; cu câteva trucuri acest baseline vă poate da nota 6
- spacy*.ipynb, [colab](https://colab.research.google.com/drive/1sbnw6BfOoneyD_gFWNCov_MN5KDVk-DM?usp=sharing) - tutorial spacy și proprietățile pe care le puteți folosi 
- transformer*.ipynb, [colab](https://colab.research.google.com/drive/1zAyi_b-OoaxoR0Bnq0Ec2-rKj1h3aXtn?usp=sharing) - tutorial huggingface transformers, cum să obții reprezentări din ultimul strat din rețea


<a name="resurse"></a> 
## Resurse permise
 
Aveți voie cu următoarele resurse externe, cu condiția să primiți aprobare de la Sergiu (aprobările se dau individual pe echipă pe mail):
- liste de cuvinte adiționale, [MRC Psycholinguistic Database](https://websites.psychology.uwa.edu.au/school/MRCDatabase/uwa_mrc.htm) conține informații privitoare la cuvinte sau lista [Dale-Chall](https://readabilityformulas.com/word-lists/the-dale-chall-word-list-for-readability-formulas/)
- seturi de date de text adiționale din care să extrageți frecvențe, de exemplu [AOCHILDES](https://github.com/UIUCLearningLanguageLab/AOCHILDES) conține ”child-directed speech transcripts, ordered by the age of the target child” pentru engleză iar Task-ul [BabyLM](https://babylm.github.io/) are ca scop antrenarea de LLM cu texte care pot fi plauzibile în dezvoltarea copiilor
- biblioteci sau API care să comunice cu [WordNet](http://wordnetweb.princeton.edu/perl/webwn?s=dog) sau [ConceptNet](https://conceptnet.io/)
- algoritmi traducere automată (ideal ar fi să ruleze local)
- rețele pre-antrenate de tip BERT, RoBERTa, XLM-RoBERTa (obligatoriu trebuie să ruleze local) din care să scoateți vectori de activări sau valori interne, word embeddings
- LLM: llama-3, mistral etc.  obligatoriu trebuie să ruleze local din care puteți scoate vectori de activări sau valori interne din rețea


**Nu aveți voie cu:**
- API-uri externe
- extragerea scorurilor de complexitate prin parsarea unor prompturi rezultate din LLM, din LLM aveți voie doar cu valori interne ale vectorilor


<a name="sugestii"></a> 
## Sugestii de caracteristici

- frecvența cuvântului într-un corpus foarte mare - aici putem folosi biblioteca wordfreq, dar ideal să implementați o funcție care extrage cuvintele frecvente dintr-un corpus arbitrar
- lungimea cuvantului
- nr de silabe
- nr de vocale (aici sunt si semivocale, fara diftongi)
- daca e titlu sau entitate
- nr de synsets din WordNet
- nr de hypernime (colour is a hypernym of red)
- nr de hyponime (spoon is a hyponym of cutlery)
- word embedding din spacy
- word embedding contextuale din modele LLM sau transformers
- nr de relații în arborele de sintaxă al propoziției
- cuvinte și părți de vorbire din context
- încercați să standardizați / normalizați datele


<a name="eval"></a> 
## Evaluarea modelelor
Evaluarea se face folosind o metrică customizată folosind două concepte din statistică:
- [Coeficientul de determinare](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Coeficientul de corelație Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)


Pentru un set de predicții: $$p = \{p_1, \cdots, p_m\}$$
și un set de valori reale care trebuiau prezise:  $$y = \{y_1, \cdots, y_m\}$$
avem coeficientul de determinare dat de mean squared error normalizată după un predictor constant:

$$
\begin{equation}
R_{py}^{2}=1-\frac{ \sum_{i}^{m}(y_i-p_i)^{2} }{ \sum_{i}^{m}(y_{i}-{\bar{y}})^{2} }
\end{equation}
$$

unde este media valorilor reale este $$\bar{y} = \frac{1}{m}\sum_{i}^{m}y$$ 

Și coeficientul de corelație Pearson:
$$\rho_{py}=\frac{\sum_{i=1}^{m}(p_{i}-{\bar{p}})(y_{i}-{\bar{y}})}{\sqrt{\sum_{i=1}^{m}{(p_{i}-{\bar{p}})}^{2}}\sqrt{\sum_{i=1}^{m}(y_{i}-{\bar{y}})^{2}}}$$
forumulat de asemenea și ca cosinusul unghiului dintre valorile standardizate (z-score) după medie și deviația standard ale predicțiilor (sau valorilor reale):

$$\frac{\sum_{i=1}^{m}(p_{i}-{\bar{p}})}{\sqrt{\sum_{i=1}^{m}{(p_{i}-{\bar{p}})}^{2}}} = \frac{p-\mu_p}{\sigma_p}$$

Coeficientul de corelație poate varia între -1, 1 și are următoarele interpretări:
- 0 când valorile nu sunt corelate
- 1 sau -1 cand valorile sunt pozitiv sau negativ corelate
- nedefinit atunci când avem varianță 0 (vector cu valori constante). 

Coeficientul de determinare poate varia între -infinit și 0 și are următoarele interpretări:
- coeficient 0 implică predicții cu valori constante egale cu media valorilor reale (o linie dreaptă care prezice de fiecare dată același număr - media)
- negativ atunci când predicțiile sunt mai proaste decât o predicție constantă cu media
- pozitiv atunci când scara predicțiilor (mean squared error) este minimizată și valorile sunt apropiate de cele reale

Metrica noastră de evaluare este dată de media dintre valoarea absolută a coeficientului de corelație și valoarea pozitivă a coeficientului de determinare:
$$\frac{1}{2}( |\rho_{py}| + \max(0, R_{py}^{2})  )  $$

<a name="data"></a> 
## Despre date
### Train
Setul de date de antrenare este format din următoarele coloane:
- id 
- language - catalan, english, filipino, french, german, italian, japanese, portuguese, sinhala, spanish
- sentence - propoziția care conține cuvântul 
- word - cuvântul căruia trebuie să-i prezicem complexitatea
- complexity - scor de complexitate pentru cuvânt



|   cur_id | language   | sentence                                                                                                                                                                                                                                                                         | word          |   complexity |
|---------:|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|-------------:|
|     8337 | catalan    | Ribó ha apel·lat tota la jerarquia eclesiàstica a Catalunya, començant pel cardenal Omella, a "trobar la màxima interlocució i col·laboració pels casos que se'ns puguin presentar" i s'ha mostrat convençut que així serà, perquè li han donat "proves que hi estan disposats". | jerarquia     |     0.525    |
|     3539 | english    | These Dhcr7 knockout mice are models for human SLOS, especially the more severely affected patients, since early postnatal lethality with respiratory failure has been reported in severe SLOS cases [4,5].                                                                      | lethality     |     0.382353 |
|     8415 | filipino   | Nilibot niya ang buong lupain at nakita ang maraming bangkay sa paligid.                                                                                                                                                                                                         | bangkay       |     0.325    |
|     8444 | french     | Je reste à votre disposition pour tout renseignement complémentaire.                                                                                                                                                                                                             | renseignement |     0.5      |
|     8466 | german     | Gleichwohl wurden auf vielen in die Westsektoren führenden Straßen, in Eisenbahnen und anderen Verkehrsmitteln durch die Volkspolizei intensiv Personenkontrollen durchgeführt, um u. a. Fluchtverdächtige und Schmuggler aufzugreifen.                                          | Schmuggler    |     0.225    |
|     8491 | italian    | Gli piaceva come raccontava il Manolo, con quelle frasi ampollose costellate di parole in dialetto                                                                                                                                                                               | parole        |     0.03     |
|     8528 | japanese   | 楽譜にはそれぞれ、長大な標題がリストによって書き添えられている。                                                                                                                                                                                                                 | 長大な        |     0.38     |
|     8561 | portuguese | Nem jurarás pela tua cabeça, porque não podes tornar um cabelo branco ou preto.                                                                                                                                                                                                  | cabelo        |     0.1      |
|     8599 | sinhala    | තායිලන්තයේ රාජකීය සඟරජ පරපුරේ ප්‍රාකෘ උඩොම් දම්මානුකුල් විබන්පෑව් නායක ස්වාමීන් වහන්සේගේ අනුශාසනා පරිදි ලුවම් පු සෘෂි කෙත් කැච් තාපසතුමා ඇතුළු තායිලන්ත දායක දායිකාවන්ගේ අනුග්‍රහයෙන් මෙම ලෝහමය ප්‍රතිමාව ඉදිකර තිබේ.                                                                                                                                        | ප්‍රතිමාව          |     0.26     |
|     8616 | spanish    | Las comunidades dependen de la producción de bienes y servicios producidos para mover la economía lo más equilibradamente posible.                                                                                                                                               | producidos    |     0.15     |

### Date imbalansate
Datele de antrenare sunt debalansate pe engleză, cu 8300 de exemple. Toate celelalte limbi au 30 de exemple de antrenare. Ideal ar fi să găsim metode de a extinde informația din engleză pe limbi noi sau de a găsi caracteristici care sunt universale pentru complexitate și deci independente de limbă.

| language   |   count |     mean |      std |   min |      25% |      50% |      75% |      max |
|:-----------|--------:|---------:|---------:|------:|---------:|---------:|---------:|---------:|
| catalan    |      30 | 0.486667 | 0.125212 | 0.175 | 0.425    | 0.475    | 0.6      | 0.7      |
| english    |    8363 | 0.301132 | 0.133718 | 0     | 0.210526 | 0.277778 | 0.368421 | 0.931818 |
| filipino   |      30 | 0.170833 | 0.125959 | 0.025 | 0.05     | 0.15     | 0.25     | 0.475    |
| french     |      30 | 0.370833 | 0.229231 | 0     | 0.16875  | 0.35     | 0.525    | 0.8      |
| german     |      30 | 0.413333 | 0.191042 | 0.15  | 0.25625  | 0.3625   | 0.55     | 0.875    |
| italian    |      30 | 0.247667 | 0.168189 | 0.03  | 0.085    | 0.235    | 0.4175   | 0.52     |
| japanese   |      30 | 0.259333 | 0.173044 | 0.02  | 0.09     | 0.27     | 0.38     | 0.62     |
| portuguese |      30 | 0.273    | 0.164572 | 0.06  | 0.13     | 0.245    | 0.38     | 0.58     |
| sinhala    |      30 | 0.243333 | 0.213966 | 0.05  | 0.09     | 0.17     | 0.3275   | 0.91     |
| spanish    |      30 | 0.449167 | 0.23346  | 0.075 | 0.23125  | 0.45     | 0.66875  | 0.875    |

### Test
Lipsește coloana complexity, pe aceea va trebui să o preziceți voi

### Date de test balansate
Pe fiecare limbă avem un număr comparabil de date de test

| language   |   sentence |
|:-----------|-----------:|
| catalan    |        445 |
| english    |        570 |
| filipino   |        570 |
| french     |        568 |
| german     |        570 |
| italian    |        570 |
| japanese   |        570 |
| portuguese |        567 |
| sinhala    |        600 |
| spanish    |        593 |



<a name="submission"></a> 
## Exemplu de submission
Pentru fiecare `cur_id` din test set, trebuie să prezici un scor de complexitate (sau o probabilitate) pentru cuvântul dat de coloana `word` și contextul `sentence`

|   cur_id |   complexity |
|---------:|-------------:|
|     8633 |  0.39513     |
|     8634 |  0.582103    |
|     8635 |  0.13757     |
|     8636 |  0.740368    |
|     8637 |  0.236702    |
|     8638 |  0.52046     |
|     8639 |  0.236432    |
|     8640 |  0.168342    |
|     8641 |  0.374964    |
|     8642 |  0.939123    |
|     8643 |  0.145224    |
|     ... |  ...    |